import type { Picture } from './types';
import { imageUrl, isVideo } from './types';

export interface MapPoint extends Picture { x: number; y: number; slot: number }
/** The way each arrow key points on the map. */
const ARROWS: Record<string, [number, number]> = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
export interface MapCamera { x: number; y: number; zoom: number }
export interface MapCallbacks {
  /** The keyboard's current image, or none. */
  onSelect: (id: string | null) => void;
  /** An image was clicked or tapped. */
  onOpen: (id: string) => void;
  /** More than one image shares a small area of the map; let the viewer choose which one to open. */
  onNearby: (ids: string[]) => void;
  /** A lifted image is being carried, at a viewport position, or was just let go (`dropped`), or the carry was
   *  cancelled (`id` null). Returns whether the position is a drop target. */
  onCarry: (id: string | null, x: number, y: number, dropped: boolean) => boolean;
  onSettle: (camera: MapCamera) => void;
}

// These mirror server/atlas.py: 64px cells holding a 60px icon, 32 per row.
const CELL = 64, INSET = 2, INNER = CELL - 2 * INSET, COLUMNS = 32;
export const PER_SHEET = COLUMNS * COLUMNS;
const CORNER = 7 / INNER;
// The play mark on a video is a disc a quarter of the icon wide, kept off icons too small for it to read.
const MARK = 0.26;
const MARK_MIN = 18;
// Holding still on an image this long lifts it, so it can be carried to the search bar; a quicker drag pans.
const HOLD = 250;
// Pointer travel that turns a press into a pan.
const SLOP = 5;
const SHARP_AT = 72;
// At seven decimal places, the smallest change in position can separate two previews by 100 CSS px.
const MAX_WORLD_SCALE = 1_000_000_000;
const MIN_ZOOM = 0.4;
const MAX_LOADS = 8;
const MAX_THUMBS = 480;
const DOT_CELL = 3;
const PREVIEW_ZOOM = 4;
const MAX_PREVIEWS = 64;
const MAX_ICON = 88;
const PREVIEW_GAP = 6;
const POINT_RADIUS = 7;

interface Camera { x: number; y: number; k: number }
interface Preview { point: number; size: number; focused: boolean }
// A sharp thumbnail, decoded at `resolution` pixels square or its own size if smaller.
interface Thumb { bitmap: ImageBitmap | null; resolution: number; loading: boolean; shown: number; used: number }

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const approach = (rate: number, dt: number) => 1 - Math.exp(-dt / rate);
// Thumbnails decode at the next power of two up from the pixels they cover, so a zoom only rarely needs a sharper one.
const resolution = (pixels: number) => 2 ** Math.ceil(Math.log2(Math.max(64, pixels)));

/** Draws the embedding on a canvas and owns the camera, hover and hit-testing. React only hears about selections, clicks and carried images. */
export class MapEngine {
  private ctx: CanvasRenderingContext2D;
  private w = 0; private h = 0; private dpr = 1;
  private points: MapPoint[] = [];
  private index = new Map<string, number>();
  private sheets: ImageBitmap[] = [];
  private cam: Camera = { x: 0, y: 0, k: 1 };
  private goal: Camera = { x: 0, y: 0, k: 1 };
  private anchor: { wx: number; wy: number; mx: number; my: number } | null = null;
  private flight = false;
  private fitK = 1; private base = 20;
  private bounds = { minX: -1, maxX: 1, minY: -1, maxY: 1 };
  private hovered = -1; private selected = -1;
  private members: Set<number> | null = null;
  private lift = new Map<number, number>();
  private highlightedIds: string[] | null = null;
  private morphing = 1; private morphFrom = new Float64Array(0); private morphTo = new Float64Array(0);
  private sx = new Float32Array(0); private sy = new Float32Array(0);
  private order: number[] = [];
  private hasAtlas = false;
  private previews: Preview[] = [];
  private dots: { context: CanvasRenderingContext2D; pixels: ImageData } | null = null;
  private thumbRequests = new AbortController();
  private thumbs = new Map<string, Thumb>();
  private loading = 0;
  private frame = 0; private last = 0; private clock = 0;
  private settle = 0;
  private pointer: { x: number; y: number } | null = null;
  private touches = new Map<number, { x: number; y: number }>();
  // A drag pans the map. Pressing and holding on an image first lifts it instead, and then the drag carries it.
  private drag: { x: number; y: number; moved: boolean; vx: number; vy: number; t: number; grabbed: number; lifted: boolean; cx: number; cy: number } | null = null;
  private hold = 0;
  private pinch: { distance: number; x: number; y: number } | null = null;
  private glide = { vx: 0, vy: 0 };
  private reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

  constructor(private canvas: HTMLCanvasElement, private stage: HTMLElement, private calls: MapCallbacks) {
    this.ctx = canvas.getContext('2d')!;
    stage.addEventListener('wheel', this.wheel, { passive: false });
    stage.addEventListener('pointerdown', this.down);
    stage.addEventListener('pointermove', this.move);
    stage.addEventListener('pointerup', this.up);
    stage.addEventListener('pointercancel', this.cancel);
    stage.addEventListener('pointerleave', this.leave);
    stage.addEventListener('dblclick', this.double);
  }

  destroy() {
    cancelAnimationFrame(this.frame); clearTimeout(this.settle); clearTimeout(this.hold);
    const s = this.stage;
    s.removeEventListener('wheel', this.wheel);
    s.removeEventListener('pointerdown', this.down);
    s.removeEventListener('pointermove', this.move);
    s.removeEventListener('pointerup', this.up);
    s.removeEventListener('pointercancel', this.cancel);
    s.removeEventListener('pointerleave', this.leave);
    s.removeEventListener('dblclick', this.double);
    this.thumbRequests.abort();
    this.thumbs.forEach(thumb => thumb.bitmap?.close()); this.thumbs.clear(); this.sheets.forEach(sheet => sheet.close()); this.sheets = [];
    this.dots = null;
  }

  // ---- Data ------------------------------------------------------------------------------------

  resize(width: number, height: number, dpr: number) {
    const first = this.w === 0, before = this.fitK;
    this.w = width; this.h = height; this.dpr = dpr;
    this.canvas.width = Math.round(width * dpr); this.canvas.height = Math.round(height * dpr);
    this.measure();
    if (first && this.points.length) this.fit(false);
    else if (this.points.length) {
      // Keep the same framing when the window changes shape.
      this.fitK = this.frameOf().k;
      const ratio = this.fitK / before;
      this.cam.k = this.clampScale(this.cam.k * ratio); this.goal.k = this.clampScale(this.goal.k * ratio);
      if (this.anchor) this.anchor = null;
    }
    this.invalidate();
  }

  /** Icon sheets by number. Sheets no point uses can be missing. */
  setSheets(sheets: ImageBitmap[]) { this.sheets.forEach(sheet => sheet.close()); this.sheets = sheets; this.invalidate(); }

  setPoints(points: MapPoint[], restore?: MapCamera) {
    const selected = this.selected >= 0 ? this.points[this.selected]?.id : null;
    this.points = points; this.morphing = 1;
    this.hasAtlas = points.some(point => point.slot >= 0);
    this.previews = [];
    this.index.clear();
    points.forEach((p, i) => this.index.set(p.id, i));
    this.sx = new Float32Array(points.length); this.sy = new Float32Array(points.length);
    this.hovered = -1; this.lift.clear(); this.members = null;
    this.selected = selected ? this.index.get(selected) ?? -1 : -1;
    if (this.selected < 0 && selected) this.calls.onSelect(null);
    this.measure();
    if (this.w) {
      this.fitK = this.frameOf().k;
      if (restore) this.jump(restore); else this.fit(false);
    }
    this.invalidate();
  }

  /** Swaps in a new arrangement of the same library: images glide from where they were to where they now belong. */
  morph(points: MapPoint[], refit: boolean) {
    if (!this.points.length || this.reduced || !this.hasAtlas || !points.some(point => point.slot >= 0)) {
      this.setPoints(points, refit ? undefined : this.camera);
      return;
    }
    const before = new Map(this.points.map(p => [p.id, p]));
    const selected = this.selected >= 0 ? this.points[this.selected].id : null;
    // The engine owns these copies: their coordinates change every frame.
    this.points = points.map(p => ({ ...p }));
    this.index = new Map(this.points.map((p, i) => [p.id, i]));
    this.sx = new Float32Array(points.length); this.sy = new Float32Array(points.length);
    this.hovered = -1; this.lift.clear(); this.members = null;
    this.selected = selected ? this.index.get(selected) ?? -1 : -1;
    if (this.selected < 0 && selected) this.calls.onSelect(null);
    this.measure();
    if (this.w) { this.fitK = this.frameOf().k; if (refit) this.fit(true); }
    this.morphFrom = new Float64Array(points.length * 2); this.morphTo = new Float64Array(points.length * 2);
    this.points.forEach((p, i) => {
      const old = before.get(p.id);
      this.morphFrom.set([old ? old.x : p.x, old ? old.y : p.y], i * 2); this.morphTo.set([p.x, p.y], i * 2);
    });
    this.morphing = 0;
    this.invalidate();
  }

  setHighlight(ids: string[] | null) {
    const set = ids && new Set(ids.map(id => this.index.get(id)).filter((i): i is number => i !== undefined));
    // Refreshing the same points (for example when an atlas arrives) keeps the user's camera.
    const previous = this.highlightedIds;
    const changed = ids !== previous && (ids?.length !== previous?.length || ids?.some((id, i) => id !== previous?.[i]));
    this.highlightedIds = ids;
    this.members = set;
    if (changed && set?.size) this.flyTo([...set]);
    this.invalidate();
  }

  /** Base icon size follows the library size, so a handful of images does not look lost. */
  private measure() {
    const n = Math.max(1, this.points.length);
    this.base = clamp(0.6 * Math.sqrt((this.w * this.h) / n), 20, 64);
    if (!this.points.length) return;
    this.bounds = this.boundsOf();
  }

  // ---- Camera ----------------------------------------------------------------------------------

  private boundsOf(indices?: number[]) {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    const count = indices?.length ?? this.points.length;
    for (let i = 0; i < count; i++) {
      const { x, y } = this.points[indices ? indices[i] : i];
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    }
    return { minX, maxX, minY, maxY };
  }

  private frameOf(indices?: number[]): Camera {
    const pad = clamp(Math.min(this.w, this.h) * 0.07, 20, 72) + this.base / 2;
    const { minX, maxX, minY, maxY } = indices ? this.boundsOf(indices) : this.bounds;
    const k = Math.min((this.w - 2 * pad) / Math.max(maxX - minX, 1e-6), (this.h - 2 * pad) / Math.max(maxY - minY, 1e-6));
    return { x: (minX + maxX) / 2, y: (minY + maxY) / 2, k };
  }

  fit(animated = true) {
    if (!this.points.length || !this.w) return;
    const frame = this.frameOf();
    this.fitK = frame.k;
    this.travel(frame, animated);
  }

  private flyTo(indices: number[]) {
    const frame = this.frameOf(indices);
    frame.k = clamp(frame.k, this.fitK * 0.8, this.fitK * 14);
    this.travel(frame, true);
  }

  private travel(goal: Camera, animated: boolean) {
    this.goal = { ...goal }; this.anchor = null; this.glide = { vx: 0, vy: 0 };
    if (animated && !this.reduced) this.flight = true; else this.cam = { ...goal };
    this.invalidate();
  }

  private clampScale(k: number) {
    return clamp(k, this.fitK * MIN_ZOOM, Math.max(this.fitK, MAX_WORLD_SCALE));
  }

  jump(camera: MapCamera) {
    const k = this.clampScale(this.fitK * camera.zoom);
    this.travel({ x: camera.x, y: camera.y, k }, false);
  }

  get camera(): MapCamera { return { x: this.goal.x, y: this.goal.y, zoom: this.goal.k / this.fitK }; }

  zoomBy(factor: number, mx = this.w / 2, my = this.h / 2, instant = false) {
    const k = this.clampScale(this.goal.k * factor);
    const wx = (mx - this.w / 2) / this.goal.k + this.goal.x, wy = (my - this.h / 2) / this.goal.k + this.goal.y;
    this.goal = { k, x: wx - (mx - this.w / 2) / k, y: wy - (my - this.h / 2) / k };
    this.flight = false; this.glide = { vx: 0, vy: 0 };
    if (instant || this.reduced) { this.cam = { ...this.goal }; this.anchor = null; } else this.anchor = { wx, wy, mx, my };
    this.limit(); this.invalidate();
  }

  panBy(dx: number, dy: number, instant = true) {
    this.goal.x += dx / this.goal.k; this.goal.y += dy / this.goal.k;
    this.anchor = null; this.flight = false;
    this.limit();
    if (instant || this.reduced) this.cam = { ...this.goal };
    this.invalidate();
  }

  /** Keep some of the data on screen, so the view can never be lost in empty space. */
  private limit() {
    const b = this.bounds, g = this.goal;
    g.x = clamp(g.x, b.minX - this.w / g.k * 0.4, b.maxX + this.w / g.k * 0.4);
    g.y = clamp(g.y, b.minY - this.h / g.k * 0.4, b.maxY + this.h / g.k * 0.4);
  }

  private iconSize(k = this.cam.k) {
    return Math.min(MAX_ICON, this.base * Math.sqrt(k / this.fitK));
  }

  // ---- Selection -------------------------------------------------------------------------------

  select(id: string | null, reveal = false) {
    const i = id === null ? -1 : this.index.get(id) ?? -1;
    if (i === this.selected) return;
    this.selected = i;
    if (i >= 0 && reveal) this.reveal(i);
    this.calls.onSelect(i >= 0 ? this.points[i].id : null);
    this.invalidate();
  }

  private reveal(i: number) {
    const p = this.points[i], g = this.goal, margin = 96;
    const x = (p.x - g.x) * g.k + this.w / 2, y = (p.y - g.y) * g.k + this.h / 2;
    if (x > margin && x < this.w - margin && y > margin && y < this.h - margin) return;
    this.goal = { ...g, x: p.x, y: p.y }; this.anchor = null; this.flight = true;
  }

  /** Move to the nearest image in a direction, within a 50 degree cone. */
  step(dx: number, dy: number) {
    const best = this.neighbour(this.selected, dx, dy);
    if (best >= 0) this.select(this.points[best].id, true);
  }

  /** Move from an image to the nearest one the way an arrow key points, for a viewer open over the map. Returns the
   *  image moved to, or null with none that way. */
  towards(id: string, key: string) {
    const best = key in ARROWS ? this.neighbour(this.index.get(id) ?? -1, ...ARROWS[key]) : -1;
    if (best < 0) return null;
    this.select(this.points[best].id, true);
    return this.points[best].id;
  }

  /** Where an image is drawn in the window, as its middle and its size, or null if the map doesn't show it. */
  place(id: string) {
    const i = this.index.get(id) ?? -1;
    if (i < 0 || (this.members && !this.members.has(i))) return null;
    const p = this.points[i], box = this.canvas.getBoundingClientRect();
    return { x: box.left + (p.x - this.cam.x) * this.cam.k + this.w / 2, y: box.top + (p.y - this.cam.y) * this.cam.k + this.h / 2, size: this.iconSize() };
  }

  private neighbour(start: number, dx: number, dy: number) {
    if (start < 0) return -1;
    const from = this.points[start];
    let best = -1, bestScore = Infinity;
    this.points.forEach((p, i) => {
      if (i === start || (this.members && !this.members.has(i))) return;
      const ax = p.x - from.x, ay = p.y - from.y, along = ax * dx + ay * dy, across = Math.abs(ax * dy - ay * dx);
      if (along <= 0 || across > along * 1.2) return;
      const score = along + across * 0.6;
      if (score < bestScore) { bestScore = score; best = i; }
    });
    return best;
  }

  /** The image nearest the middle of the view. */
  nearestToCenter() {
    let best = -1, bestDistance = Infinity;
    const c = this.goal;
    this.points.forEach((p, i) => {
      if (this.members && !this.members.has(i)) return;
      const d = (p.x - c.x) ** 2 + (p.y - c.y) ** 2;
      if (d < bestDistance) { bestDistance = d; best = i; }
    });
    return best >= 0 ? this.points[best].id : null;
  }

  // ---- Pointer input ---------------------------------------------------------------------------

  private local(e: { clientX: number; clientY: number }) {
    const r = this.canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }
  private onChrome(e: Event) { return !!(e.target as Element).closest?.('.map-controls, .map-nearby, button, a'); }

  private wheel = (e: WheelEvent) => {
    if (this.onChrome(e)) return;
    e.preventDefault();
    const { x, y } = this.local(e);
    const delta = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1);
    this.zoomBy(Math.exp(-delta * (e.ctrlKey ? 0.012 : 0.002)), x, y);
  };

  private down = (e: PointerEvent) => {
    if (this.onChrome(e) || (e.pointerType === 'mouse' && e.button !== 0)) return;
    this.stage.setPointerCapture(e.pointerId);
    const p = this.local(e);
    this.touches.set(e.pointerId, p);
    this.glide = { vx: 0, vy: 0 };
    clearTimeout(this.hold);
    if (this.touches.size === 1) {
      const d = this.drag = { ...p, moved: false, vx: 0, vy: 0, t: e.timeStamp, grabbed: this.hit(p.x, p.y), lifted: false, cx: e.clientX, cy: e.clientY };
      if (d.grabbed >= 0) this.hold = setTimeout(() => this.pickUp(), HOLD);
    } else { this.dropCarry(); this.drag = null; this.startPinch(); }
  };

  /** The held image comes up under the pointer; from here the drag carries it and the map stays still. */
  private pickUp() {
    const d = this.drag;
    if (!d || d.moved || d.grabbed < 0) return;
    d.lifted = true;
    this.canvas.style.cursor = 'grabbing';
    this.setHover(-1);
    this.calls.onCarry(this.points[d.grabbed].id, d.cx, d.cy, false);
  }

  /** Puts a carried image back without dropping it anywhere. */
  private dropCarry() {
    if (this.drag?.lifted) this.calls.onCarry(null, 0, 0, false);
  }

  private startPinch() {
    const [a, b] = [...this.touches.values()];
    this.pinch = { distance: Math.hypot(a.x - b.x, a.y - b.y) || 1, x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  }

  private move = (e: PointerEvent) => {
    if (!this.drag && this.onChrome(e)) return;
    const p = this.local(e);
    if (this.touches.has(e.pointerId)) this.touches.set(e.pointerId, p);
    if (this.pinch && this.touches.size >= 2) {
      const [a, b] = [...this.touches.values()];
      const distance = Math.hypot(a.x - b.x, a.y - b.y) || 1, x = (a.x + b.x) / 2, y = (a.y + b.y) / 2;
      this.panBy(-(x - this.pinch.x), -(y - this.pinch.y));
      this.zoomBy(distance / this.pinch.distance, x, y, true);
      this.pinch = { distance, x, y };
      return;
    }
    if (this.drag && this.touches.has(e.pointerId)) {
      const d = this.drag, dx = p.x - d.x, dy = p.y - d.y;
      d.cx = e.clientX; d.cy = e.clientY;
      if (d.lifted) { this.calls.onCarry(this.points[d.grabbed].id, e.clientX, e.clientY, false); return; }
      if (!d.moved && Math.hypot(dx, dy) < SLOP) return;
      d.moved = true; this.canvas.style.cursor = 'grabbing'; clearTimeout(this.hold);
      const dt = Math.max(1, e.timeStamp - d.t);
      d.vx = 0.7 * (dx / dt) + 0.3 * d.vx; d.vy = 0.7 * (dy / dt) + 0.3 * d.vy;
      d.x = p.x; d.y = p.y; d.t = e.timeStamp;
      this.panBy(-dx, -dy);
      this.setHover(-1);
      return;
    }
    this.pointer = p;
    this.setHover(this.hit(p.x, p.y));
  };

  private up = (e: PointerEvent) => {
    const p = this.local(e), d = this.drag;
    this.touches.delete(e.pointerId);
    if (this.touches.size < 2) this.pinch = null;
    if (!d) return;
    clearTimeout(this.hold);
    this.drag = null; this.canvas.style.cursor = '';
    if (d.lifted) {
      // Let go anywhere: on the search bar it finds similar images, elsewhere the image just goes back.
      this.calls.onCarry(this.points[d.grabbed].id, e.clientX, e.clientY, true);
    } else if (!d.moved) {
      const i = this.hit(p.x, p.y);
      if (i >= 0) this.openPoint(i); else this.select(null);
    } else if (!this.reduced && e.timeStamp - d.t < 80 && Math.hypot(d.vx, d.vy) > 0.05) {
      this.glide = { vx: d.vx, vy: d.vy }; this.invalidate();
    }
    if (e.pointerType !== 'mouse') this.setHover(-1);
  };

  private cancel = (e: PointerEvent) => {
    clearTimeout(this.hold); this.dropCarry();
    this.touches.delete(e.pointerId); this.pinch = null; this.drag = null; this.canvas.style.cursor = '';
  };
  private leave = () => { this.pointer = null; if (!this.drag) this.setHover(-1); };
  private double = (e: MouseEvent) => {
    if (this.onChrome(e)) return;
    const { x, y } = this.local(e);
    this.zoomBy(2.4, x, y);
  };

  // ---- Hit testing and hover -------------------------------------------------------------------

  /** Prefer an image's point center, even when another image's preview overlaps it. */
  private hit(x: number, y: number) {
    let closest = -1, distance = POINT_RADIUS ** 2;
    for (let n = this.order.length - 1; n >= 0; n--) {
      const i = this.order[n], d = (this.sx[i] - x) ** 2 + (this.sy[i] - y) ** 2;
      if (d < distance) { closest = i; distance = d; }
    }
    if (closest >= 0) return closest;
    for (let n = this.previews.length - 1; n >= 0; n--) {
      const { point, size } = this.previews[n];
      if (Math.abs(x - this.sx[point]) <= size / 2 && Math.abs(y - this.sy[point]) <= size / 2) return point;
    }
    return -1;
  }

  openSelected() {
    if (this.selected >= 0) this.openPoint(this.selected);
  }

  private openPoint(point: number) {
    const x = this.sx[point], y = this.sy[point];
    const nearby = [this.points[point].id];
    // Topmost first after the selected image, so highlighted images retain their existing priority.
    for (let n = this.order.length - 1; n >= 0; n--) {
      const i = this.order[n];
      if (i !== point && (this.sx[i] - x) ** 2 + (this.sy[i] - y) ** 2 <= POINT_RADIUS ** 2) nearby.push(this.points[i].id);
    }
    if (nearby.length > 1) this.calls.onNearby(nearby);
    else this.calls.onOpen(nearby[0]);
  }

  private setHover(i: number) {
    if (i === this.hovered) return;
    this.hovered = i;
    if (i >= 0 && !this.lift.has(i)) this.lift.set(i, 0);
    this.canvas.style.cursor = i >= 0 ? 'pointer' : '';
    this.invalidate();
  }

  private liftedSize(i: number, size: number) { const t = this.lift.get(i) ?? 0; return size + Math.max(18, size * 0.35) * t; }

  // ---- Frame loop ------------------------------------------------------------------------------

  invalidate() { if (!this.frame) this.frame = requestAnimationFrame(this.tick); }

  private tick = (now: number) => {
    this.frame = 0;
    const dt = Math.min(48, this.last ? now - this.last : 16); this.last = now; this.clock = now;
    const busy = this.update(dt);
    this.draw();
    if (busy || this.pendingThumbs) this.invalidate(); else { this.last = 0; this.settled(); }
  };

  private update(dt: number) {
    let busy = false;
    const { cam, goal } = this;
    if (this.glide.vx || this.glide.vy) {
      this.panBy(-this.glide.vx * dt, -this.glide.vy * dt);
      const decay = Math.exp(-dt / 320);
      this.glide.vx *= decay; this.glide.vy *= decay;
      if (Math.hypot(this.glide.vx, this.glide.vy) < 0.01) this.glide = { vx: 0, vy: 0 };
      busy = true;
    }
    const a = approach(this.flight ? 130 : 80, dt);
    const dk = Math.log(goal.k / cam.k);
    if (Math.abs(dk) > 0.0004) { cam.k *= Math.exp(dk * a); busy = true; } else cam.k = goal.k;
    if (this.anchor) {
      const n = this.anchor;
      cam.x = n.wx - (n.mx - this.w / 2) / cam.k; cam.y = n.wy - (n.my - this.h / 2) / cam.k;
      if (cam.k === goal.k) this.anchor = null;
    } else {
      const ex = goal.x - cam.x, ey = goal.y - cam.y;
      if (Math.abs(ex) * cam.k > 0.2 || Math.abs(ey) * cam.k > 0.2) { cam.x += ex * a; cam.y += ey * a; busy = true; } else { cam.x = goal.x; cam.y = goal.y; this.flight = false; }
    }
    if (this.anchor) busy = true;
    if (this.morphing < 1) {
      this.morphing = Math.min(1, this.morphing + dt / 700);
      const eased = 1 - Math.pow(1 - this.morphing, 3), { morphFrom: a, morphTo: b } = this;
      this.points.forEach((p, i) => { p.x = a[i * 2] + (b[i * 2] - a[i * 2]) * eased; p.y = a[i * 2 + 1] + (b[i * 2 + 1] - a[i * 2 + 1]) * eased; });
      busy = true;
    }
    for (const [i, t] of this.lift) {
      const goalLift = i === this.hovered ? 1 : 0;
      const next = this.reduced ? goalLift : t + (goalLift - t) * approach(70, dt);
      if (goalLift === 0 && next < 0.01) this.lift.delete(i); else { this.lift.set(i, Math.abs(goalLift - next) < 0.005 ? goalLift : next); if (this.lift.get(i) !== goalLift) busy = true; }
    }
    if (this.pointer && !this.drag && busy) this.setHover(this.hit(this.pointer.x, this.pointer.y));
    return busy;
  }

  private settled() {
    clearTimeout(this.settle);
    // Before any points arrive the camera means nothing; saving it would override the first fit.
    if (!this.points.length) return;
    this.settle = window.setTimeout(() => this.calls.onSettle(this.camera), 350);
  }

  // ---- Drawing ---------------------------------------------------------------------------------

  private pendingThumbs = false;

  private draw() {
    const { ctx, cam, w, h, dpr, points } = this;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    // Set again by any thumbnail still fading in or waiting to load, which keeps frames coming.
    this.pendingThumbs = false;
    const size = this.iconSize();
    for (let i = points.length - 1; i >= 0; i--) {
      this.sx[i] = (points[i].x - cam.x) * cam.k + w / 2; this.sy[i] = (points[i].y - cam.y) * cam.k + h / 2;
    }
    this.order = this.stack(size / 2);
    this.drawPreviews(size);
    this.drawPoints(new Set(this.previews.map(preview => preview.point)));
  }

  /** Draw a bounded set of previews with the same spacing for cached thumbnails and atlas icons. */
  private drawPreviews(size: number) {
    const { ctx, w, h } = this;
    const arranged: Preview[] = [];
    const place = (point: number, previewSize: number, focused: boolean) => {
      const x = this.sx[point], y = this.sy[point];
      if (arranged.some(other => {
        const gap = (previewSize + other.size) / 2 + PREVIEW_GAP;
        return Math.abs(x - this.sx[other.point]) < gap && Math.abs(y - this.sy[other.point]) < gap;
      })) return;
      arranged.push({ point, size: previewSize, focused });
    };
    const focus = [this.hovered, this.selected].filter((i, n, all) =>
      i >= 0 && all.indexOf(i) === n && this.sx[i] >= 0 && this.sx[i] <= w && this.sy[i] >= 0 && this.sy[i] <= h);
    for (const i of focus) place(i, Math.max(36, this.liftedSize(i, size)), true);

    if (this.hasAtlas || this.cam.k / this.fitK >= PREVIEW_ZOOM) {
      // Pick one candidate per cell, then check its exact bounds against the cards already placed.
      const candidates: number[] = [], occupied = new Set<number>();
      const cell = Math.max(32, size * 0.8), stride = Math.ceil(w / cell) + 1;
      for (let n = this.order.length - 1; n >= 0; n--) {
        const i = this.order[n], x = this.sx[i], y = this.sy[i];
        if (x < 0 || x > w || y < 0 || y > h || focus.includes(i) || (this.members && !this.members.has(i))) continue;
        const bin = Math.floor(y / cell) * stride + Math.floor(x / cell);
        if (!occupied.has(bin)) { occupied.add(bin); candidates.push(i); }
      }
      const distance = (i: number) => (this.sx[i] - w / 2) ** 2 + (this.sy[i] - h / 2) ** 2;
      candidates.sort((a, b) => distance(a) - distance(b));
      for (const i of candidates) {
        if (arranged.length >= MAX_PREVIEWS) break;
        place(i, size, false);
      }
    }
    // Hovered and selected points load immediately. Wait until the camera stops before loading its new viewport.
    const arrived = this.cam.k === this.goal.k && this.cam.x === this.goal.x && this.cam.y === this.goal.y && !this.drag;
    for (const preview of arranged) {
      if (this.loading >= MAX_LOADS) break;
      if (!arrived && !preview.focused) continue;
      const point = this.points[preview.point];
      if ((!this.sheet(point) || preview.size * this.dpr > SHARP_AT)
        && this.outgrown(this.thumbs.get(point.id), preview.size)) this.load(point.id, preview.size);
    }
    this.previews = arranged.reverse().filter(({ point, focused }) =>
      focused || this.sheet(this.points[point]) || this.thumbs.get(this.points[point].id)?.bitmap);
    for (const { point, size: previewSize, focused } of this.previews) {
      this.tile(ctx, point, previewSize, focused ? this.lift.get(point) ?? 0 : 0);
    }
  }

  /** Draw points not represented by a preview above the cards so even a covered point stays visible. */
  private drawPoints(represented: Set<number>) {
    const { ctx, w, h } = this;
    const columns = Math.ceil(w / DOT_CELL), rows = Math.ceil(h / DOT_CELL);
    if (!columns || !rows) return;
    if (this.dots?.pixels.width !== columns || this.dots.pixels.height !== rows) {
      const canvas = document.createElement('canvas');
      canvas.width = columns; canvas.height = rows;
      const context = canvas.getContext('2d')!;
      this.dots = { context, pixels: context.createImageData(columns, rows) };
    }
    const { context, pixels } = this.dots;
    const rgba = pixels.data, dark = document.documentElement.dataset.theme === 'dark';
    const neutral = dark ? 202 : 71;
    rgba.fill(0);
    for (let n = this.order.length - 1; n >= 0; n--) {
      const i = this.order[n], x = Math.floor(this.sx[i] / DOT_CELL), y = Math.floor(this.sy[i] / DOT_CELL);
      if (x < 0 || x >= columns || y < 0 || y >= rows || represented.has(i)) continue;
      const offset = (y * columns + x) * 4;
      if (rgba[offset + 3]) continue;
      const member = this.members?.has(i);
      rgba[offset] = member ? (dark ? 137 : 40) : neutral;
      rgba[offset + 1] = member ? (dark ? 182 : 101) : neutral;
      rgba[offset + 2] = member ? (dark ? 255 : 202) : neutral;
      rgba[offset + 3] = member ? 245 : this.members ? 55 : 175;
    }
    context.putImageData(pixels, 0, 0);
    ctx.globalAlpha = 1;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(context.canvas, 0, 0, columns * DOT_CELL, rows * DOT_CELL);
    ctx.imageSmoothingEnabled = true;
  }

  /** Visible images, bottom to top: faded ones under lit ones. */
  private stack(half: number) {
    const lit: number[] = [], faded: number[] = [], lo = -half, right = this.w + half, bottom = this.h + half;
    for (let i = this.points.length - 1; i >= 0; i--) {
      const x = this.sx[i], y = this.sy[i];
      if (x < lo || x > right || y < lo || y > bottom) continue;
      (!this.members || this.members.has(i) ? lit : faded).push(i);
    }
    return faded.concat(lit);
  }

  private sheet(point: MapPoint) {
    return this.sheets[Math.floor(point.slot / PER_SHEET)];
  }

  private tile(ctx: CanvasRenderingContext2D, i: number, size: number, lift: number) {
    const p = this.points[i], sheet = this.sheet(p), thumb = this.thumbs.get(p.id);
    const cell = p.slot % PER_SHEET, x = this.sx[i] - size / 2, y = this.sy[i] - size / 2;
    ctx.globalAlpha = 1;
    if (lift > 0) { ctx.shadowColor = 'rgba(0,0,0,.22)'; ctx.shadowBlur = 24 * lift; ctx.shadowOffsetY = 8 * lift; }
    if (sheet) ctx.drawImage(sheet, (cell % COLUMNS) * CELL + INSET, Math.floor(cell / COLUMNS) * CELL + INSET, INNER, INNER, x, y, size, size);
    else if (i === this.selected || i === this.hovered) {
      ctx.fillStyle = 'rgba(110,117,128,.68)';
      ctx.beginPath(); ctx.roundRect(x, y, size, size, size * CORNER); ctx.fill();
    }
    if (lift > 0) { ctx.shadowColor = 'transparent'; ctx.shadowBlur = 0; ctx.shadowOffsetY = 0; }
    if (thumb?.bitmap) {
      thumb.used = this.clock;
      const fade = clamp((this.clock - thumb.shown) / 200, 0, 1);
      ctx.save();
      ctx.beginPath(); ctx.roundRect(x, y, size, size, size * CORNER); ctx.clip();
      ctx.globalAlpha = fade;
      ctx.drawImage(thumb.bitmap, x, y, size, size);
      ctx.restore();
      if (fade < 1) this.pendingThumbs = true;
    }
    if (isVideo(p) && size >= MARK_MIN) this.playMark(ctx, x, y, size);
  }

  /** A dark disc with a white triangle in the corner, readable over any photograph in either theme. Drawn last, so
   *  the sharp thumbnail never covers it, and at the icon's alpha, so it recedes with the icon. */
  private playMark(ctx: CanvasRenderingContext2D, x: number, y: number, size: number) {
    const r = size * MARK / 2, cx = x + size * CORNER + r * 0.4, cy = y + size - size * CORNER - r * 0.4;
    ctx.fillStyle = 'rgba(0,0,0,.5)';
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();
    // The triangle's centre of mass sits left of its bounding box, so it is nudged right to look centred.
    const t = r * 0.42, left = cx - t * 0.6;
    ctx.fillStyle = '#fff';
    ctx.beginPath(); ctx.moveTo(left, cy - t); ctx.lineTo(left + t * 1.7, cy); ctx.lineTo(left, cy + t); ctx.closePath(); ctx.fill();
  }

  // ---- Sharp thumbnails ------------------------------------------------------------------------

  private outgrown(thumb: Thumb | undefined, size: number) { return !thumb || (!thumb.loading && thumb.resolution < resolution(size * this.dpr)); }

  private load(id: string, size: number) {
    if (!this.thumbs.has(id) && this.thumbs.size >= MAX_THUMBS) {
      let oldest: string | undefined, used = Infinity;
      for (const [key, thumb] of this.thumbs) {
        if (!thumb.loading && thumb.used < used) { oldest = key; used = thumb.used; }
      }
      if (oldest === undefined) return;
      this.thumbs.get(oldest)?.bitmap?.close();
      this.thumbs.delete(oldest);
    }
    const entry = this.thumbs.get(id) ?? { bitmap: null, resolution: 0, loading: false, shown: 0, used: this.clock };
    const wanted = entry.resolution = resolution(size * this.dpr);
    entry.loading = true; this.thumbs.set(id, entry); this.loading++;
    // Decoded off the main thread, cropped to the square the map shows at about the size it is drawn, and kept decoded:
    // a canvas would otherwise decode the full thumbnail mid-frame, and again whenever the browser discarded it.
    fetch(imageUrl(id), { signal: this.thumbRequests.signal }).then(response => {
      if (!response.ok) throw new Error('Preview unavailable');
      return response.blob();
    }).then(createImageBitmap).then(async full => {
      const side = Math.min(full.width, full.height), target = Math.min(side, wanted);
      let bitmap: ImageBitmap;
      try {
        bitmap = await createImageBitmap(full, (full.width - side) / 2, (full.height - side) / 2, side, side, { resizeWidth: target, resizeHeight: target, resizeQuality: 'high' });
      } finally {
        full.close();
      }
      if (this.thumbs.get(id) !== entry) { bitmap.close(); return; }
      if (!entry.bitmap) entry.shown = performance.now();
      entry.bitmap?.close(); entry.bitmap = bitmap; this.invalidate();
    }).catch(() => {}).finally(() => {
      entry.loading = false;
      this.loading--;
      if (this.thumbs.get(id) === entry) this.invalidate();
    });
  }

  // ---- Keyboard --------------------------------------------------------------------------------

  key(e: KeyboardEvent) {
    if (e.metaKey || e.ctrlKey || e.altKey) return false;
    if (e.key in ARROWS) {
      const [dx, dy] = ARROWS[e.key];
      if (this.selected >= 0 && !e.shiftKey) this.step(dx, dy); else this.panBy(dx * 120, dy * 120, false);
    } else if (e.key === '+' || e.key === '=') this.zoomBy(1.6);
    else if (e.key === '-' || e.key === '_') this.zoomBy(1 / 1.6);
    else if (e.key === '0') this.fit();
    else if (e.key === 'Escape' && this.selected >= 0) this.select(null);
    else if ((e.key === 'Enter' || e.key === ' ') && this.selected < 0) { const id = this.nearestToCenter(); if (id) this.select(id, true); }
    else return false;
    return true;
  }
}
