use std::{fmt::Debug, hash::Hash};

use egui::{Color32, Context, FontId, Id, Key, Pos2, Rect, Sense, Stroke, StrokeKind, Ui, UiBuilder, Vec2, pos2, vec2};

use crate::{
    layout::{ResizeHandle, ResizeHandleStyle, SplitAxis, SplitSize, split_rect},
    tabs,
};

use super::{Action, DockError, DropTarget, Layout, Node, Permissions, Placement, resolve_drop};

/// Application-supplied chrome. Body renderers retain their own UI style and contents.
#[derive(Clone, Debug)]
pub struct DockStyle {
    pub tab_height: f32,
    pub gap: f32,
    pub min_pane: f32,
    pub background: Color32,
    pub tab_background: Color32,
    pub active_background: Color32,
    pub text: Color32,
    pub inactive_text: Color32,
    pub border: Stroke,
    pub accent: Color32,
    pub font: FontId,
    /// Localize these action labels in the application adapter.
    pub float_label: String,
    pub close_label: String,
    pub move_label: String,
    pub panels_label: String,
    pub resize_label: String,
    pub resize_window_label: String,
}

impl DockStyle {
    pub fn from_ui(ui: &Ui) -> Self {
        let visuals = ui.visuals();
        Self {
            tab_height: 26.0,
            gap: 4.0,
            min_pane: 40.0,
            background: visuals.panel_fill,
            tab_background: visuals.extreme_bg_color,
            active_background: visuals.widgets.active.bg_fill,
            text: visuals.text_color(),
            inactive_text: visuals.weak_text_color(),
            border: visuals.widgets.noninteractive.bg_stroke,
            accent: visuals.selection.stroke.color,
            font: FontId::proportional(12.0),
            float_label: "Float panel".into(),
            close_label: "Close panel".into(),
            move_label: "Move to group".into(),
            panels_label: "Panels".into(),
            resize_label: "Resize panels".into(),
            resize_window_label: "Resize panel window".into(),
        }
    }

    fn valid(&self) -> bool {
        [self.tab_height, self.gap, self.min_pane, self.border.width, self.font.size].iter().all(|v| v.is_finite() && (0.0..=4096.0).contains(v))
            && self.tab_height > 0.0
            && self.font.size > 0.0
    }
}

#[derive(Clone, Debug)]
pub struct DockOutput<P> {
    pub actions: Vec<Action<P>>,
    pub error: Option<DockError>,
    pub focused: Option<P>,
    /// Whole-floating-group gesture identity; host may display its own group drop preview.
    pub group_drag: Option<P>,
    /// Whole-group release for host-owned atomic docking policy.
    pub group_drop: Option<(P, Pos2)>,
}

impl<P> Default for DockOutput<P> {
    fn default() -> Self {
        Self { actions: Vec::new(), error: None, focused: None, group_drag: None, group_drop: None }
    }
}

/// Preferred body bounds. Tiny viewports relax minima proportionally rather than producing
/// inverted rectangles. Maximums constrain a pane when its sibling can occupy the remainder.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PanelLimits {
    pub min: Vec2,
    pub max: Vec2,
}

impl Default for PanelLimits {
    fn default() -> Self {
        Self { min: Vec2::ZERO, max: Vec2::splat(1_000_000.0) }
    }
}

impl PanelLimits {
    fn valid(self) -> bool {
        self.min.is_finite()
            && self.max.is_finite()
            && self.min.x >= 0.0
            && self.min.y >= 0.0
            && self.max.x >= self.min.x
            && self.max.y >= self.min.y
            && self.max.x <= 1_000_000.0
            && self.max.y <= 1_000_000.0
    }
}

#[derive(Clone)]
struct Drag<P> {
    panel: P,
    origin: Pos2,
    source: Rect,
    grab: Vec2,
}

#[derive(Clone)]
struct GroupDrag<P> {
    keep_cursor: bool,
    origin: Pos2,
    panel: P,
    original: [f32; 4],
}

/// Start a docking drag from an application's existing tab/header. Use the same area ID when
/// switching from a legacy layout to [`DockArea`]. The model is untouched until a valid release.
pub fn begin_drag<P: Clone + Send + Sync + 'static>(ctx: &Context, area_id: Id, panel: P, source_rect: Rect) -> bool {
    let Some(origin) = ctx.pointer_interact_pos().filter(|p| p.is_finite()) else { return false };
    if !super::targets::usable(source_rect) {
        return false;
    }
    let grab = (origin - source_rect.min).clamp(Vec2::ZERO, source_rect.size());
    ctx.data_mut(|d| d.insert_temp(area_id.with("drag"), Drag { panel, origin, source: source_rect, grab }));
    true
}

/// Discard pending tab/group gestures and an actively dragged renderer-owned control before
/// replacing a workspace. Body/custom-header/menu gestures and other areas are untouched.
/// Ownership is recorded as controls render; before an initial render or drag threshold there
/// may be no active egui drag to cancel. Keyboard focus is unchanged. The host owns the
/// replacement snapshot, so this does not restore a group's previous rectangle.
pub fn cancel_drag<P: Clone + Send + Sync + 'static>(ctx: &Context, area_id: Id) {
    let dragged = ctx.dragged_id();
    let owned = ctx.data_mut(|d| {
        d.remove::<Drag<P>>(area_id.with("drag"));
        d.remove::<GroupDrag<P>>(area_id.with("group-drag"));
        let ids = d.remove_temp::<Vec<Id>>(area_id.with("owned-interactions"));
        dragged.is_some_and(|id| ids.is_some_and(|ids| ids.contains(&id)))
    });
    if owned {
        ctx.stop_dragging();
    }
}

/// Host-owned drop resolution. Returning feedback consumes the gesture, including no-op
/// targets. Hosts commit their staged transaction on release; the renderer emits no move.
pub struct DropRequest<'a, P> {
    pub panel: &'a P,
    pub pointer: Pos2,
    pub targets: &'a [DropTarget<P>],
    pub floating_rect: Option<Rect>,
    pub whole_group: bool,
    pub released: bool,
    pub workspace: Rect,
    pub screen: Rect,
}

#[derive(Clone, Debug, Default)]
pub struct DropFeedback {
    pub preview: Option<Rect>,
    pub insertion: Option<Rect>,
}

/// One mutable application adapter owns bodies, menus and shaped text, avoiding competing
/// mutable closures. Logical labels remain separate and are used for tab accessibility.
pub trait DockContent<P> {
    fn body(&mut self, ui: &mut Ui, panel: &P);
    fn resolve_drag(&mut self, _ctx: &Context, _style: &DockStyle, _request: &DropRequest<'_, P>) -> Option<DropFeedback> {
        None
    }

    fn label(&mut self, _ui: &Ui, _panel: &P, logical: &str, _font: &FontId, _color: Color32) -> egui::WidgetText {
        logical.into()
    }
    /// Shape stock Float/Move/Close action text independently of panel identity.
    /// The renderer preserves `logical` for accessibility, even when the visual text differs.
    fn action_label(&mut self, _ui: &Ui, logical: &str, _font: &FontId, _color: Color32) -> egui::WidgetText {
        logical.into()
    }
    /// Full rendered group bounds, including its tab header.
    /// Opt into transient UI dragging without moving the saved source layout.
    fn live_drag(&self) -> bool {
        false
    }
    fn drag_members(&self, _ctx: &Context, _panel: &P) -> Option<Vec<P>> {
        None
    }
    fn render_drag_ui(&self) -> bool {
        true
    }
    /// Preferred floating size, independent of the current dock geometry.
    fn floating_size(&self, _panel: &P) -> Option<Vec2> {
        None
    }
    /// Publish measured targets for hosts coordinating multiple docking surfaces.
    fn drop_targets(&mut self, _ctx: &Context, _targets: &[DropTarget<P>]) {}
    fn group_rect(&mut self, _ctx: &Context, _panels: &[P], _rect: Rect) {}
    fn panel_menu(&mut self, _ui: &mut Ui, _panel: &P) {}
    fn group_header_width(&self, _panels: &[P], _active: &P, _floating: bool) -> f32 {
        0.0
    }
    fn group_header(&mut self, _ui: &mut Ui, _panels: &[P], _active: &P, _floating: bool) {}
}

struct BodyContent<F>(F);
impl<P, F: FnMut(&mut Ui, &P)> DockContent<P> for BodyContent<F> {
    fn body(&mut self, ui: &mut Ui, panel: &P) {
        (self.0)(ui, panel);
    }
}

/// Optional renderer for applications without an existing docking surface. Apps with custom
/// chrome can instead use the model, target resolver and resize/tab primitives independently.
pub struct DockArea<P = ()> {
    id: Id,
    surface: Id,
    activation: Option<P>,
    blockers: Vec<Rect>,
}

impl<P: Clone + Eq + Hash + Debug + Send + Sync + 'static> DockArea<P> {
    pub fn new(id: Id) -> Self {
        Self { id, surface: id, activation: None, blockers: Vec::new() }
    }

    /// Independent divider identity for multiple surfaces sharing a drag session.
    pub fn surface(mut self, id: Id) -> Self {
        self.surface = id;
        self
    }

    /// Whether keyboard focus belongs to a rendered docking control in this layout.
    /// Panel bodies and application-provided header/menu widgets are excluded. Hosts may
    /// route nonconflicting global shortcuts through chrome while retaining widget ownership
    /// of activation, traversal, resize and cancellation keys. Invalid layouts return false.
    pub fn chrome_has_focus(&self, ctx: &egui::Context, layout: &Layout<P>) -> bool {
        let Some(focused) = ctx.memory(|memory| memory.focused()) else { return false };
        if layout.validate().is_err() || ctx.read_response(focused).is_none() {
            return false;
        }
        let matches = |id: Id| id == focused;
        let group = |panels: &[P]| {
            panels.iter().any(|panel| matches(self.id.with(("tab", panel))) || matches(self.id.with(("overflow", panel))))
                || ["move-group", "move-group-header", "move-group-border"].iter().any(|name| matches(self.id.with((*name, panels.first()))))
        };
        for floating in &layout.floating {
            if group(&floating.panels)
                || floating.panels.first().is_some_and(|first| {
                    ["resize", "resize-left", "resize-right", "resize-bottom", "resize-bottom-left"]
                        .iter()
                        .any(|name| matches(self.id.with(("floating", first)).with(*name)))
                })
            {
                return true;
            }
        }
        let mut nodes = layout.root.as_ref().map(|root| vec![(root, Vec::<bool>::new())]).unwrap_or_default();
        while let Some((node, path)) = nodes.pop() {
            match node {
                Node::Tabs { panels, .. } if group(panels) => return true,
                Node::Stack { entries } => {
                    if entries.iter().any(|entry| {
                        matches(self.id.with(("tab", &entry.panel)))
                            || matches(self.id.with(("collapse", &entry.panel)))
                            || matches(self.id.with(("stack-resize", &entry.panel)))
                    }) {
                        return true;
                    }
                }
                Node::Split { first, second, .. } => {
                    if matches(self.id.with(("split", &path))) {
                        return true;
                    }
                    let mut second_path = path.clone();
                    second_path.push(true);
                    let mut first_path = path;
                    first_path.push(false);
                    nodes.push((second, second_path));
                    nodes.push((first, first_path));
                }
                _ => {}
            }
        }
        false
    }

    /// Raise and focus a panel's owning floating area on this frame. Apply Activate through
    /// the application's command path as well to select its tab.
    pub fn raise_panel(mut self, panel: P) -> Self {
        self.activation = Some(panel);
        self
    }

    /// Host overlays that must reject docking gestures, including floating fallback.
    pub fn blocked_rects(mut self, rects: &[Rect]) -> Self {
        self.blockers = rects.to_vec();
        self
    }

    /// Render a validated snapshot and return requested actions for the app's command handler.
    /// No action is applied here. Panel contents get a global stable egui ID across moves.
    /// Floating groups are egui areas inside this viewport; this does not create native windows.
    pub fn show(
        self,
        ui: &mut Ui,
        layout: &Layout<P>,
        style: &DockStyle,
        label: impl Fn(&P) -> String,
        permissions: impl Fn(&P) -> Permissions,
        body: impl FnMut(&mut Ui, &P),
    ) -> DockOutput<P> {
        self.show_with_limits(ui, layout, style, label, permissions, |_| PanelLimits::default(), body)
    }

    /// Like [`Self::show`], with per-panel size policies supplied by the application.
    #[allow(clippy::too_many_arguments)]
    pub fn show_with_limits(
        self,
        ui: &mut Ui,
        layout: &Layout<P>,
        style: &DockStyle,
        label: impl Fn(&P) -> String,
        permissions: impl Fn(&P) -> Permissions,
        limits: impl Fn(&P) -> PanelLimits,
        body: impl FnMut(&mut Ui, &P),
    ) -> DockOutput<P> {
        self.show_customized(ui, layout, style, label, permissions, limits, &mut BodyContent(body))
    }

    /// Render with application-owned chrome and shaped labels. Header callbacks queue app
    /// commands; structural changes are applied after rendering, like DockOutput actions.
    #[allow(clippy::too_many_arguments)]
    pub fn show_customized(
        self,
        ui: &mut Ui,
        layout: &Layout<P>,
        style: &DockStyle,
        label: impl Fn(&P) -> String,
        permissions: impl Fn(&P) -> Permissions,
        limits: impl Fn(&P) -> PanelLimits,
        content: &mut impl DockContent<P>,
    ) -> DockOutput<P> {
        let rect = ui.available_rect_before_wrap();
        if let Err(error) = layout.validate() {
            cancel_drag::<P>(ui.ctx(), self.id);
            return DockOutput { error: Some(error), ..Default::default() };
        }
        if !style.valid() || !super::targets::usable(rect) || layout.panels().into_iter().any(|p| !limits(p).valid()) {
            cancel_drag::<P>(ui.ctx(), self.id);
            return DockOutput { error: Some(DockError::InvalidGeometry), ..Default::default() };
        }
        let mut renderer = Renderer {
            id: self.id,
            surface: self.surface,
            style,
            label: &label,
            permissions: &permissions,
            limits: &limits,
            content,
            activation: self.activation,
            blockers: self.blockers,
            layout,
            targets: Vec::new(),
            owned_interactions: Vec::new(),
            output: DockOutput::default(),
            workspace: rect,
        };
        if let Some(root) = &layout.root {
            renderer.node(ui, root, rect, &mut Vec::new());
            ui.allocate_rect(rect, Sense::hover());
        }
        renderer.floating(ui.ctx());
        renderer.content.drop_targets(ui.ctx(), &renderer.targets);
        renderer.drag_ui(ui.ctx());
        renderer.finish_group_drag(ui.ctx());
        renderer.finish_drag(ui.ctx());
        ui.ctx().data_mut(|d| d.insert_temp(self.id.with("owned-interactions"), renderer.owned_interactions));
        renderer.output
    }
}

struct Renderer<'a, P> {
    workspace: Rect,
    id: Id,
    surface: Id,
    style: &'a DockStyle,
    label: &'a dyn Fn(&P) -> String,
    permissions: &'a dyn Fn(&P) -> Permissions,
    limits: &'a dyn Fn(&P) -> PanelLimits,
    content: &'a mut dyn DockContent<P>,
    activation: Option<P>,
    blockers: Vec<Rect>,
    layout: &'a Layout<P>,
    targets: Vec<DropTarget<P>>,
    owned_interactions: Vec<Id>,
    output: DockOutput<P>,
}

impl<P: Clone + Eq + Hash + Debug + Send + Sync + 'static> Renderer<'_, P> {
    fn register_interaction(&mut self, ctx: &Context, id: Id) {
        // A validated tree has at most 2*MAX_PANELS nodes. Tabs, stack dividers,
        // floating grips/corners and splits fit comfortably inside this fixed bound.
        const CAP: usize = 16 * super::MAX_PANELS;
        if self.owned_interactions.len() < CAP && !self.owned_interactions.contains(&id) {
            self.owned_interactions.push(id);
        }
        // Retain prior-frame controls until return, and publish current controls before
        // app callbacks. No Context lock spans a callback or a stop_dragging call.
        ctx.data_mut(|d| {
            let ids = d.get_temp_mut_or_default::<Vec<Id>>(self.id.with("owned-interactions"));
            if ids.len() < CAP && !ids.contains(&id) {
                ids.push(id);
            }
        });
    }

    fn node(&mut self, ui: &mut Ui, node: &Node<P>, rect: Rect, path: &mut Vec<bool>) {
        if !super::targets::usable(rect) {
            return;
        }
        match node {
            Node::Split { axis, size, first, second } => {
                let Some((split, lo, hi)) = constrained_split(rect, *axis, *size, self.style.gap, self.node_limits(first), self.node_limits(second))
                else {
                    return;
                };
                path.push(false);
                self.node(ui, first, split.first, path);
                path.pop();
                path.push(true);
                self.node(ui, second, split.second, path);
                path.pop();
                // The divider's inner-edge hit margin must win over child scrollbars, just
                // as accordion dividers win over the adjacent panel's content.
                let response = ResizeHandle::new(
                    self.surface.with(("split", &*path)),
                    *axis,
                    split.divider,
                    &self.style.resize_label,
                    ResizeHandleStyle { hovered: self.style.accent, dragged: self.style.accent, focused: self.style.accent },
                )
                .value(split.first_size(), lo..=hi)
                .show(ui);
                self.register_interaction(ui.ctx(), response.response.id);
                if response.delta != 0.0
                    && let Some(size) = split.resized(*size, (split.first_size() + response.delta).clamp(lo, hi) - split.first_size(), 0.0)
                {
                    self.output.actions.push(Action::ResizeSplit { path: path.clone(), size });
                }
            }
            Node::Tabs { panels, active } => self.group(ui, panels, *active, rect, false),
            Node::Stack { entries } => {
                let total_header = (entries.len() as f32 * self.style.tab_height).min(rect.height());
                let remaining = (rect.height() - total_header).max(0.0);
                let total_wanted: f64 = entries.iter().filter(|e| e.open).map(|e| f64::from(e.height.unwrap_or(self.style.min_pane).max(1.0))).sum();
                let header = self.style.tab_height.min(rect.height() / entries.len().max(1) as f32);
                let mut top = rect.top();
                let mut dividers = Vec::new();
                for (index, entry) in entries.iter().enumerate() {
                    let share = if entry.open && total_wanted > 0.0 {
                        (f64::from(remaining) * f64::from(entry.height.unwrap_or(self.style.min_pane).max(1.0)) / total_wanted) as f32
                    } else {
                        0.0
                    };
                    let panel_rect = Rect::from_min_max(pos2(rect.left(), top), pos2(rect.right(), (top + header + share).min(rect.bottom())));
                    let title_rect = Rect::from_min_max(panel_rect.min, pos2(panel_rect.right(), (top + header).min(panel_rect.bottom())));
                    self.tab(ui, &entry.panel, title_rect, entry.open, panel_rect);
                    let collapse = Rect::from_min_max(pos2((title_rect.right() - 18.0).max(title_rect.left()), title_rect.top()), title_rect.max);
                    let response = ui.interact(collapse, self.id.with(("collapse", &entry.panel)), Sense::click());
                    response.widget_info(|| {
                        egui::WidgetInfo::selected(egui::WidgetType::Checkbox, ui.is_enabled(), entry.open, (self.label)(&entry.panel))
                    });
                    ui.painter().with_clip_rect(collapse).text(
                        collapse.center(),
                        egui::Align2::CENTER_CENTER,
                        if entry.open { "−" } else { "+" },
                        self.style.font.clone(),
                        self.style.text,
                    );
                    if response.clicked() {
                        self.output.actions.push(Action::SetStackOpen { panel: entry.panel.clone(), open: !entry.open });
                    }
                    let body_rect = Rect::from_min_max(pos2(rect.left(), title_rect.bottom()), panel_rect.max);
                    if entry.open {
                        self.contents(ui, &entry.panel, body_rect);
                    }
                    self.targets.push(DropTarget {
                        anchor: entry.panel.clone(),
                        rect: panel_rect,
                        tab_strip: Some(title_rect),
                        tabs: vec![(entry.panel.clone(), title_rect)],
                        allow_splits: true,
                        accepts_tabs: (self.permissions)(&entry.panel).accepts_tabs,
                    });
                    top = panel_rect.bottom();
                    if entry.open && remaining > 0.0 {
                        dividers.push((index, Rect::from_min_max(pos2(rect.left(), top - 1.0), pos2(rect.right(), top + 1.0)).intersect(rect)));
                    }
                }
                // Register dividers after the headers so their narrow hit regions win at the
                // shared boundary. Closed entries retain their saved sizes and are skipped.
                for (index, divider) in dividers {
                    let Some(first) = entries.get(index) else { continue };
                    let Some(second) = entries.iter().skip(index.saturating_add(1)).find(|entry| entry.open) else { continue };
                    let a = f64::from(first.height.unwrap_or(self.style.min_pane).max(1.0));
                    let b = f64::from(second.height.unwrap_or(self.style.min_pane).max(1.0));
                    let scale = f64::from(remaining) / total_wanted;
                    let first_size = (a * scale) as f32;
                    let pair_size = ((a + b) * scale) as f32;
                    let body_limits = |panel: &P| {
                        let mut limits = (self.limits)(panel);
                        limits.min.y = limits.min.y.max(self.style.min_pane);
                        limits.max = limits.max.max(limits.min);
                        limits
                    };
                    let (lo, hi) = split_bounds(pair_size, SplitAxis::Vertical, body_limits(&first.panel), body_limits(&second.panel));
                    let response = ResizeHandle::new(
                        self.id.with(("stack-resize", &first.panel)),
                        SplitAxis::Vertical,
                        divider,
                        &self.style.resize_label,
                        ResizeHandleStyle { hovered: self.style.accent, dragged: self.style.accent, focused: self.style.accent },
                    )
                    .value(first_size, lo..=hi)
                    .show(ui);
                    self.register_interaction(ui.ctx(), response.response.id);
                    if response.delta != 0.0 && scale > 0.0 {
                        let delta = f64::from((first_size + response.delta).clamp(lo, hi) - first_size) / scale;
                        let next_a = (a + delta).clamp((a + b - 1_000_000.0).max(1.0), (a + b - 1.0).min(1_000_000.0));
                        self.output.actions.push(Action::ResizeStack { panel: first.panel.clone(), height: Some(next_a as f32) });
                        self.output.actions.push(Action::ResizeStack { panel: second.panel.clone(), height: Some((a + b - next_a) as f32) });
                    }
                }
            }
        }
    }

    fn node_limits(&self, node: &Node<P>) -> PanelLimits {
        node_limits(node, self.style, &self.permissions, &self.limits)
    }

    fn contents(&mut self, ui: &mut Ui, panel: &P, rect: Rect) {
        if !super::targets::usable(rect) {
            return;
        }
        ui.painter().rect_filled(rect, 0.0, self.style.background);
        let moving =
            self.content.live_drag()
                && (ui.ctx().data(|d| d.get_temp::<Drag<P>>(self.id.with("drag"))).is_some_and(|drag| {
                    self.content.drag_members(ui.ctx(), &drag.panel).unwrap_or_else(|| vec![drag.panel.clone()]).contains(panel)
                }) || ui
                    .ctx()
                    .data(|d| d.get_temp::<GroupDrag<P>>(self.id.with("group-drag")))
                    .is_some_and(|drag| self.layout.floating.iter().any(|g| g.panels.contains(&drag.panel) && g.panels.contains(panel))));
        if moving {
            return;
        }
        let mut child =
            ui.new_child(UiBuilder::new().id(self.id.with(("body", panel))).max_rect(rect).layout(egui::Layout::top_down(egui::Align::Min)));
        child.set_clip_rect(rect.intersect(ui.clip_rect()));
        self.content.body(&mut child, panel);
        if child.rect_contains_pointer(rect) && child.input(|i| i.pointer.primary_pressed()) {
            self.output.focused = Some(panel.clone());
        }
    }

    fn move_floating_group(&mut self, ui: &mut Ui, panels: &[P], rect: Rect, hit: Rect, id: Id) {
        if !super::targets::usable(hit) {
            return;
        }
        let response = ui.interact(hit, id, Sense::drag());
        let response = if id == self.id.with(("move-group-border", panels.first())) {
            response.on_hover_cursor(egui::CursorIcon::Default)
        } else {
            response.on_hover_cursor(egui::CursorIcon::Grab)
        };
        self.register_interaction(ui.ctx(), response.id);
        response.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), &self.style.move_label));
        if panels.iter().all(|p| (self.permissions)(p).movable)
            && response.drag_started()
            && let Some(first) = panels.first()
        {
            let origin = ui.input(|input| input.pointer.press_origin()).unwrap_or(hit.center());
            ui.ctx().data_mut(|d| {
                d.insert_temp(
                    self.id.with("group-drag"),
                    GroupDrag {
                        keep_cursor: id == self.id.with(("move-group-border", panels.first())),
                        panel: first.clone(),
                        original: [rect.left(), rect.top(), rect.width(), rect.height()],
                        origin,
                    },
                )
            });
        }
        if panels.iter().all(|p| (self.permissions)(p).movable)
            && response.dragged()
            && let Some(drag) = ui.ctx().data(|d| d.get_temp::<GroupDrag<P>>(self.id.with("group-drag")))
            && let Some(pointer) = ui.input(|input| input.pointer.interact_pos())
            && !ui.input(|i| i.key_pressed(Key::Escape) || !i.focused)
        {
            let [x, y, width, height] = drag.original;
            // Resolve from the original press, avoiding cumulative pixel rounding as the
            // Area follows fractional pointer steps.
            let moved = clamp_float(Rect::from_min_size(pos2(x, y), vec2(width, height)).translate(pointer - drag.origin), ui.ctx().content_rect());
            if !self.content.live_drag()
                && let Some(first) = panels.first()
            {
                self.output
                    .actions
                    .push(Action::MoveFloating { panel: first.clone(), rect: [moved.left(), moved.top(), moved.width(), moved.height()] });
            }
        }
    }

    fn group(&mut self, ui: &mut Ui, panels: &[P], active: usize, rect: Rect, floating: bool) {
        let Some(selected) = panels.get(active) else { return };
        self.content.group_rect(ui.ctx(), panels, rect);
        let tab_drag = ui.ctx().data(|d| d.get_temp::<Drag<P>>(self.id.with("drag")));
        let group_drag = ui.ctx().data(|d| d.get_temp::<GroupDrag<P>>(self.id.with("group-drag")));
        let whole = self.content.live_drag()
            && (group_drag.is_some_and(|d| panels.contains(&d.panel))
                || tab_drag.is_some_and(|d| panels.contains(&d.panel) && self.content.drag_members(ui.ctx(), &d.panel).is_some_and(|m| m.len() > 1)));
        if whole {
            ui.painter().rect_filled(rect, 0.0, self.style.background);
            return;
        }

        let policy = (self.permissions)(selected);
        let bare = panels.len() == 1 && policy == Permissions::PROTECTED;
        if bare {
            self.contents(ui, selected, rect);
            self.targets.push(DropTarget {
                anchor: selected.clone(),
                rect,
                tab_strip: None,
                tabs: Vec::new(),
                allow_splits: !floating,
                accepts_tabs: false,
            });
            return;
        }
        let mut strip = Rect::from_min_max(rect.min, pos2(rect.right(), (rect.top() + self.style.tab_height).min(rect.bottom())));
        ui.painter().rect_filled(strip, 0.0, self.style.tab_background);
        if floating {
            let grip = Rect::from_min_max(strip.min, pos2((strip.left() + 12.0).min(strip.right()), strip.bottom()));
            self.move_floating_group(ui, panels, rect, grip, self.id.with(("move-group", panels.first())));
            ui.painter().line_segment([grip.center() - vec2(0.0, 5.0), grip.center() + vec2(0.0, 5.0)], self.style.border);
            strip.min.x = grip.right();
        }
        let requested = self.content.group_header_width(panels, selected, floating);
        let header_width = if requested.is_finite() { requested.clamp(0.0, strip.width()) } else { 0.0 };
        if header_width > 0.0 {
            let header = Rect::from_min_max(pos2(strip.right() - header_width, strip.top()), strip.max);
            let mut child = ui.new_child(UiBuilder::new().id(self.id.with(("header", selected))).max_rect(header));
            child.set_clip_rect(header.intersect(ui.clip_rect()));
            self.content.group_header(&mut child, panels, selected, floating);
        }
        let widths: Vec<f32> = panels
            .iter()
            .map(|p| {
                self.content
                    .label(ui, p, &(self.label)(p), &self.style.font, self.style.text)
                    .into_galley(ui, Some(egui::TextWrapMode::Extend), f32::INFINITY, self.style.font.clone())
                    .size()
                    .x
                    + 24.0
            })
            .collect();
        let fit = tabs::fit(&widths, active, (strip.width() - header_width).max(0.0), 40.0, 22.0);
        let mut x = strip.left();
        let mut targets = Vec::new();
        for (index, width) in &fit.shown {
            let Some(panel) = panels.get(*index) else { continue };
            let tab = Rect::from_min_size(pos2(x, strip.top()), vec2(*width, strip.height()));
            self.tab(ui, panel, tab, *index == active, rect);
            targets.push((panel.clone(), tab));
            x += width;
        }
        if !fit.overflow.is_empty() && fit.overflow_width > 0.0 {
            let overflow = Rect::from_min_size(pos2(x, strip.top()), vec2(fit.overflow_width, strip.height()));
            let response = ui.interact(overflow, self.id.with(("overflow", selected)), Sense::click());
            response.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), &self.style.panels_label));
            ui.painter().with_clip_rect(overflow).text(overflow.center(), egui::Align2::CENTER_CENTER, "…", self.style.font.clone(), self.style.text);
            egui::Popup::menu(&response).show(|ui| {
                egui::ScrollArea::vertical().max_height(360.0).show(ui, |ui| {
                    for index in &fit.overflow {
                        if let Some(panel) = panels.get(*index)
                            && self.menu_label(ui, panel).clicked()
                        {
                            self.output.actions.push(Action::Activate { panel: panel.clone() });
                            self.output.focused = Some(panel.clone());
                            ui.close();
                        }
                    }
                });
            });
        }
        if !fit.overflow.is_empty() {
            x += fit.overflow_width;
        }
        if floating {
            let blank = Rect::from_min_max(pos2(x, strip.top()), pos2((strip.right() - header_width).max(x), strip.bottom()));
            self.move_floating_group(ui, panels, rect, blank, self.id.with(("move-group-header", panels.first())));
            let border = Rect::from_min_max(rect.min, pos2(rect.right(), (rect.top() + 4.0).min(strip.bottom())));
            self.move_floating_group(ui, panels, rect, border, self.id.with(("move-group-border", panels.first())));
        }
        // Retain logical order even when overflow hides a tab; hidden identities have no hit area.
        let targets = panels
            .iter()
            .map(|panel| {
                let rect = targets.iter().find(|(p, _)| p == panel).map(|(_, rect)| *rect).unwrap_or(Rect::NOTHING);
                (panel.clone(), rect)
            })
            .collect();
        self.targets.push(DropTarget {
            anchor: selected.clone(),
            rect,
            tab_strip: Some(strip),
            tabs: targets,
            allow_splits: !floating,
            accepts_tabs: policy.accepts_tabs,
        });
        self.contents(ui, selected, Rect::from_min_max(pos2(rect.left(), strip.bottom()), rect.max));
        ui.painter().rect_stroke(rect, 0.0, self.style.border, StrokeKind::Inside);
    }

    fn menu_label(&mut self, ui: &mut Ui, panel: &P) -> egui::Response {
        let logical = (self.label)(panel);
        let text = self.content.label(ui, panel, &logical, &self.style.font, self.style.text);
        let response = ui.button(text);
        response.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), &logical));
        response
    }

    fn tab(&mut self, ui: &mut Ui, panel: &P, rect: Rect, active: bool, group: Rect) {
        let policy = (self.permissions)(panel);
        let label = (self.label)(panel);
        let color = if active { self.style.text } else { self.style.inactive_text };
        let galley = self.content.label(ui, panel, &label, &self.style.font, color).into_galley(
            ui,
            Some(egui::TextWrapMode::Extend),
            f32::INFINITY,
            self.style.font.clone(),
        );
        let response = tabs::Tab::new(self.id.with(("tab", panel)), &label, active)
            .sense(if policy.movable { Sense::click_and_drag() } else { Sense::click() })
            .focus_stroke(Stroke::new(1.0, self.style.accent))
            .show_at(ui, rect, |ui, response| {
                let painter = ui.painter().with_clip_rect(rect.intersect(ui.clip_rect()));
                painter.rect_filled(rect, 0.0, if active || response.hovered() { self.style.active_background } else { self.style.tab_background });
                painter.galley(rect.center() - galley.size() * 0.5, galley.clone(), color);
            });
        self.register_interaction(ui.ctx(), response.id);
        if response.clicked() {
            self.output.actions.push(Action::Activate { panel: panel.clone() });
            self.output.focused = Some(panel.clone());
        }
        if policy.movable && response.drag_started() {
            begin_drag(ui.ctx(), self.id, panel.clone(), group);
        }
        if self.content.live_drag() && ui.ctx().data(|d| d.get_temp::<Drag<P>>(self.id.with("drag"))).is_some_and(|d| &d.panel == panel) {
            ui.painter().with_clip_rect(rect.intersect(ui.clip_rect())).rect_filled(rect, 0.0, self.style.tab_background);
        }
        response.context_menu(|ui| {
            self.content.panel_menu(ui, panel);
            if policy.movable && policy.floatable {
                let text = self.content.action_label(ui, &self.style.float_label, &self.style.font, self.style.text);
                let response = ui.button(text);
                response.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), &self.style.float_label));
                if response.clicked() {
                    let size = self.content.floating_size(panel).filter(|s| s.is_finite() && s.x > 0.0 && s.y > 0.0).unwrap_or(group.size());
                    let rect = clamp_float(Rect::from_min_size(group.min + vec2(24.0, 24.0), size), ui.ctx().content_rect());
                    self.output.actions.push(Action::Float { panel: panel.clone(), rect: [rect.left(), rect.top(), rect.width(), rect.height()] });
                    ui.close();
                }
            }
            if policy.movable {
                let text = self.content.action_label(ui, &self.style.move_label, &self.style.font, self.style.text);
                let response = ui.menu_button(text, |ui| {
                    egui::ScrollArea::vertical().max_height(360.0).show(ui, |ui| {
                        for target in self.layout.panels() {
                            if target != panel && (self.permissions)(target).accepts_tabs && self.menu_label(ui, target).clicked() {
                                self.output.actions.push(Action::Move {
                                    panel: panel.clone(),
                                    anchor: target.clone(),
                                    placement: Placement::Tab { before: None },
                                });
                                ui.close();
                            }
                        }
                    });
                });
                response.response.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), &self.style.move_label));
            }
            if policy.closable {
                let text = self.content.action_label(ui, &self.style.close_label, &self.style.font, self.style.text);
                let response = ui.button(text);
                response.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), &self.style.close_label));
                if response.clicked() {
                    self.output.actions.push(Action::Close { panel: panel.clone() });
                    ui.close();
                }
            }
        });
    }

    fn floating(&mut self, ctx: &Context) {
        let screen = ctx.content_rect();
        if !super::targets::usable(screen) {
            return;
        }
        let layers = ctx.memory(|memory| memory.layer_ids().collect::<Vec<_>>());
        let mut groups: Vec<_> = self.layout.floating.iter().collect();
        groups.sort_by_key(|group| {
            let id = group.panels.first().map(|panel| self.id.with(("floating", panel)));
            layers.iter().position(|layer| Some(layer.id) == id).unwrap_or(usize::MAX)
        });
        for group in groups {
            let Some(first) = group.panels.first() else { continue };
            let [x, y, width, height] = group.rect;
            let limits = self.node_limits(&Node::Tabs { panels: group.panels.clone(), active: group.active });
            let size = vec2(width, height).max(limits.min).min(limits.max);
            let rect = clamp_float(Rect::from_min_size(pos2(x, y), size), screen);
            let id = self.id.with(("floating", first));
            if self.activation.as_ref().is_some_and(|panel| group.panels.contains(panel)) {
                ctx.move_to_top(egui::LayerId::new(egui::Order::Middle, id));
                ctx.memory_mut(|memory| memory.request_focus(self.id.with(("tab", self.activation.as_ref().unwrap_or(first)))));
                self.output.focused = self.activation.clone();
            }
            egui::Area::new(id).order(egui::Order::Middle).fixed_pos(rect.min).movable(false).constrain_to(screen).show(ctx, |ui| {
                ui.set_min_size(rect.size());
                let rect = Rect::from_min_size(ui.min_rect().min, rect.size());
                ui.allocate_rect(rect, Sense::hover());
                self.group(ui, &group.panels, group.active, rect, true);
                let corners = [
                    ("resize", Rect::from_min_max((rect.max - vec2(12.0, 12.0)).max(rect.min), rect.max), vec2(1.0, 1.0)),
                    (
                        "resize-left",
                        Rect::from_min_max(pos2(rect.left(), rect.top() + 12.0), pos2((rect.left() + 4.0).min(rect.right()), rect.bottom() - 12.0)),
                        vec2(-1.0, 0.0),
                    ),
                    (
                        "resize-right",
                        Rect::from_min_max(pos2((rect.right() - 4.0).max(rect.left()), rect.top() + 12.0), pos2(rect.right(), rect.bottom() - 12.0)),
                        vec2(1.0, 0.0),
                    ),
                    (
                        "resize-bottom",
                        Rect::from_min_max(pos2(rect.left() + 12.0, (rect.bottom() - 4.0).max(rect.top())), pos2(rect.right() - 12.0, rect.bottom())),
                        vec2(0.0, 1.0),
                    ),
                    (
                        "resize-bottom-left",
                        Rect::from_min_max(
                            pos2(rect.left(), (rect.bottom() - 12.0).max(rect.top())),
                            pos2((rect.left() + 12.0).min(rect.right()), rect.bottom()),
                        ),
                        vec2(-1.0, 1.0),
                    ),
                ];
                for (name, corner, direction) in corners {
                    if !super::targets::usable(corner) {
                        continue;
                    }
                    let cursor = if direction.x == 0.0 {
                        egui::CursorIcon::ResizeVertical
                    } else if direction.y == 0.0 {
                        egui::CursorIcon::ResizeHorizontal
                    } else if direction.x == direction.y {
                        egui::CursorIcon::ResizeNwSe
                    } else {
                        egui::CursorIcon::ResizeNeSw
                    };
                    let response = ui.interact(corner, id.with(name), Sense::drag()).on_hover_cursor(cursor);
                    self.register_interaction(ui.ctx(), response.id);
                    response.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), &self.style.resize_window_label));
                    if response.drag_started() {
                        response.request_focus();
                    }
                    let mut delta = if response.dragged() { ui.input(|i| i.pointer.delta()) } else { Vec2::ZERO };
                    if response.has_focus() {
                        ui.memory_mut(|memory| {
                            memory.set_focus_lock_filter(
                                response.id,
                                egui::EventFilter { horizontal_arrows: true, vertical_arrows: true, ..Default::default() },
                            )
                        });
                        ui.input_mut(|input| {
                            if input.consume_key(egui::Modifiers::NONE, Key::ArrowLeft) {
                                delta.x -= 8.0;
                            }
                            if input.consume_key(egui::Modifiers::NONE, Key::ArrowRight) {
                                delta.x += 8.0;
                            }
                            if input.consume_key(egui::Modifiers::NONE, Key::ArrowUp) {
                                delta.y -= 8.0;
                            }
                            if input.consume_key(egui::Modifiers::NONE, Key::ArrowDown) {
                                delta.y += 8.0;
                            }
                        });
                    }
                    if response.enabled()
                        && group.panels.iter().all(|panel| (self.permissions)(panel).movable)
                        && delta != Vec2::ZERO
                        && !ui.input(|input| input.key_pressed(Key::Escape) || !input.focused)
                    {
                        let size = (rect.size() + delta * direction)
                            .max(limits.min.max(vec2(40.0, self.style.tab_height + 20.0)))
                            .min(limits.max)
                            .min(screen.size());
                        let min = pos2(
                            if direction.x < 0.0 { rect.right() - size.x } else { rect.left() },
                            if direction.y < 0.0 { rect.bottom() - size.y } else { rect.top() },
                        );
                        let resized = clamp_float(Rect::from_min_size(min, size), screen);
                        self.output.actions.push(Action::MoveFloating {
                            panel: first.clone(),
                            rect: [resized.left(), resized.top(), resized.width(), resized.height()],
                        });
                    }
                    if name == "resize" && (response.hovered() || response.dragged() || response.has_focus()) {
                        ui.painter().line_segment([corner.left_bottom(), corner.right_top()], Stroke::new(2.0, self.style.accent));
                    }
                }
            });
        }
    }

    fn drag_ui(&mut self, ctx: &Context) {
        if !self.content.live_drag()
            || !self.content.render_drag_ui()
            || ctx.input(|i| !i.focused || i.key_pressed(Key::Escape) || !i.pointer.primary_down())
        {
            return;
        }
        let Some(pointer) = ctx.pointer_interact_pos() else {
            return;
        };
        let tab = ctx.data(|d| d.get_temp::<Drag<P>>(self.id.with("drag")));
        let group = ctx.data(|d| d.get_temp::<GroupDrag<P>>(self.id.with("group-drag")));
        let (panels, selected, rect) = if let Some(drag) = tab {
            let size = self.content.floating_size(&drag.panel).unwrap_or(drag.source.size());
            let grab = drag.grab.min(vec2(size.x, self.style.tab_height));
            (
                self.content.drag_members(ctx, &drag.panel).unwrap_or_else(|| vec![drag.panel.clone()]),
                drag.panel,
                clamp_float(Rect::from_min_size(pointer - grab, size), ctx.content_rect()),
            )
        } else if let Some(drag) = group {
            let Some(g) = self.layout.floating.iter().find(|g| g.panels.contains(&drag.panel)) else {
                return;
            };
            let [x, y, w, h] = drag.original;
            (
                g.panels.clone(),
                g.panels.get(g.active).unwrap_or(&drag.panel).clone(),
                clamp_float(Rect::from_min_size(pos2(x, y), vec2(w, h)).translate(pointer - drag.origin), ctx.content_rect()),
            )
        } else {
            return;
        };
        egui::Area::new(self.id.with("drag-ui")).order(egui::Order::Foreground).fixed_pos(rect.min).movable(false).interactable(false).show(
            ctx,
            |ui| {
                ui.set_min_size(rect.size());
                ui.disable();
                ui.set_clip_rect(rect);
                ui.painter().rect_filled(rect, 0.0, self.style.background);
                let header = Rect::from_min_size(rect.min, vec2(rect.width(), self.style.tab_height.min(rect.height())));
                ui.painter().rect_filled(header, 0.0, self.style.tab_background);
                let mut x = header.left();
                for panel in panels {
                    let text = (self.label)(&panel);
                    let galley = ui.painter().layout_no_wrap(text, self.style.font.clone(), self.style.text);
                    let width = (galley.size().x + 20.0).min((header.right() - x).max(0.0));
                    let tab = Rect::from_min_size(pos2(x, header.top()), vec2(width, header.height()));
                    if panel == selected {
                        ui.painter().rect_filled(tab, 0.0, self.style.active_background);
                    }
                    ui.painter().with_clip_rect(tab).galley(pos2(x + 10.0, header.center().y - galley.size().y / 2.0), galley, self.style.text);
                    x += width;
                }
                let body = Rect::from_min_max(pos2(rect.left(), header.bottom()), rect.max);
                let mut child = ui.new_child(UiBuilder::new().id(self.id.with(("body", &selected))).max_rect(body));
                child.set_clip_rect(body);
                self.content.body(&mut child, &selected);
                ui.painter().rect_stroke(rect, 0.0, self.style.border, StrokeKind::Inside);
            },
        );
    }

    fn host_drop(&mut self, ctx: &Context, panel: &P, pointer: Pos2, whole_group: bool, released: bool, floating_rect: Option<Rect>) -> bool {
        let request = DropRequest {
            panel,
            pointer,
            targets: &self.targets,
            floating_rect,
            whole_group,
            released,
            workspace: self.workspace,
            screen: ctx.content_rect(),
        };
        let Some(feedback) = self.content.resolve_drag(ctx, self.style, &request) else { return false };
        if !released {
            let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, self.id.with("drop-preview")));
            if let Some(rect) = feedback.preview.filter(|rect| super::targets::usable(*rect)) {
                painter.rect(rect, 0.0, self.style.accent.gamma_multiply(0.15), Stroke::new(2.0, self.style.accent), StrokeKind::Inside);
            }
            if let Some(rect) = feedback.insertion.filter(|rect| super::targets::usable(*rect)) {
                painter.rect_filled(rect, 0.0, self.style.accent);
            }
        }
        true
    }

    fn finish_group_drag(&mut self, ctx: &Context) {
        let Some(drag) = ctx.data(|d| d.get_temp::<GroupDrag<P>>(self.id.with("group-drag"))) else { return };
        if !self.layout.floating.iter().any(|g| g.panels.contains(&drag.panel) && g.panels.iter().all(|p| (self.permissions)(p).movable)) {
            if !self.content.live_drag() {
                ctx.data_mut(|d| d.remove::<GroupDrag<P>>(self.id.with("group-drag")));
            }
            return;
        }
        if drag.keep_cursor {
            ctx.set_cursor_icon(egui::CursorIcon::Default);
        }
        let (escape, focused, down, released, pointer) =
            ctx.input(|i| (i.key_pressed(Key::Escape), i.focused, i.pointer.primary_down(), i.pointer.primary_released(), i.pointer.interact_pos()));
        if !escape
            && focused
            && (down || released)
            && let Some(pointer) = pointer.filter(|p| p.is_finite())
            && !self.blockers.iter().any(|r| r.contains(pointer))
            && self.host_drop(
                ctx,
                &drag.panel,
                pointer,
                true,
                released,
                self.content.live_drag().then(|| {
                    let [x, y, w, h] = drag.original;
                    clamp_float(Rect::from_min_size(pos2(x, y), vec2(w, h)).translate(pointer - drag.origin), ctx.content_rect())
                }),
            )
        {
            if released {
                ctx.data_mut(|d| d.remove::<GroupDrag<P>>(self.id.with("group-drag")));
            }
            return;
        }
        if escape || !focused || !down {
            ctx.data_mut(|d| d.remove::<GroupDrag<P>>(self.id.with("group-drag")));
            if escape || !focused {
                if !self.content.live_drag() {
                    self.output.actions.push(Action::MoveFloating { panel: drag.panel, rect: drag.original });
                }
            } else if released
                && let Some(pointer) = pointer.filter(|p| p.is_finite())
                && !self.blockers.iter().any(|r| r.contains(pointer))
            {
                self.output.group_drop = Some((drag.panel, pointer));
            }
        } else if pointer.is_some_and(|p| !self.blockers.iter().any(|r| r.contains(p))) {
            self.output.group_drag = Some(drag.panel);
        }
    }

    fn tab_float_rect(&self, ctx: &Context, drag: &Drag<P>, pointer: Pos2) -> Rect {
        let size = self.content.floating_size(&drag.panel).filter(|s| s.is_finite() && s.x > 0.0 && s.y > 0.0).unwrap_or(drag.source.size());
        let grab = drag.grab.min(vec2(size.x, self.style.tab_height));
        clamp_float(Rect::from_min_size(pointer - grab, size), ctx.content_rect())
    }

    fn finish_drag(&mut self, ctx: &Context) {
        let Some(drag) = ctx.data(|d| d.get_temp::<Drag<P>>(self.id.with("drag"))) else { return };
        let (pointer, down, released, focused) =
            ctx.input(|i| (i.pointer.interact_pos(), i.pointer.primary_down(), i.pointer.primary_released(), i.focused));
        let escape = ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, Key::Escape));
        if escape || !focused || (!self.content.live_drag() && !self.layout.contains(&drag.panel)) || (!down && !released) {
            cancel_drag::<P>(ctx, self.id);
            return;
        }
        let Some(pointer) = pointer.filter(|p| p.is_finite()) else {
            if released {
                cancel_drag::<P>(ctx, self.id);
            }
            return;
        };
        if self.blockers.iter().any(|rect| rect.contains(pointer)) {
            if released {
                cancel_drag::<P>(ctx, self.id);
            }
            return;
        }
        let floating_rect = ((self.permissions)(&drag.panel).floatable
            && (!drag.source.contains(pointer) || self.layout.floating.iter().any(|g| g.panels.contains(&drag.panel))))
        .then(|| self.tab_float_rect(ctx, &drag, pointer));
        if self.host_drop(ctx, &drag.panel, pointer, false, released, floating_rect) {
            if released {
                cancel_drag::<P>(ctx, self.id);
            }
            return;
        }
        let proposal = resolve_drop(&drag.panel, pointer, &self.targets);
        if released && proposal.as_ref().is_some_and(|proposal| proposal.noop) {
            if released {
                cancel_drag::<P>(ctx, self.id);
            }
            return;
        }
        if released {
            cancel_drag::<P>(ctx, self.id);
            if pointer.distance(drag.origin) < 2.0 {
                return;
            }
            let action = if let Some(proposal) = proposal {
                Some(Action::Move { panel: drag.panel.clone(), anchor: proposal.anchor, placement: proposal.placement })
            } else if (self.permissions)(&drag.panel).floatable
                && (!drag.source.contains(pointer) || self.layout.floating.iter().any(|g| g.panels.contains(&drag.panel)))
            {
                let rect = self.tab_float_rect(ctx, &drag, pointer);
                Some(Action::Float { panel: drag.panel.clone(), rect: [rect.left(), rect.top(), rect.width(), rect.height()] })
            } else {
                None
            };
            if let Some(action) = action {
                // Validate the proposed gesture against the same complete snapshot. The app
                // still applies it via its own command path after body rendering finishes.
                let mut candidate = self.layout.clone();
                if candidate.apply_with_permissions(action.clone(), self.permissions).is_ok() {
                    self.output.actions.push(action);
                    self.output.focused = Some(drag.panel);
                }
            }
            return;
        }
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, self.id.with("drop-preview")));
        if let Some(proposal) = proposal {
            let same_group = proposal.insertion.is_some()
                && self.targets.iter().any(|target| target.rect == proposal.preview && target.tabs.iter().any(|(panel, _)| panel == &drag.panel));
            if !same_group {
                painter.rect(
                    proposal.preview.shrink(1.0),
                    0.0,
                    self.style.accent.gamma_multiply(0.15),
                    Stroke::new(2.0, self.style.accent),
                    StrokeKind::Inside,
                );
            }
            if let Some(marker) = proposal.insertion {
                painter.rect_filled(marker, 0.0, self.style.accent);
            }
        } else if (self.permissions)(&drag.panel).floatable {
            let rect = self.tab_float_rect(ctx, &drag, pointer);
            painter.rect_stroke(rect, 0.0, Stroke::new(2.0, self.style.accent), StrokeKind::Inside);
        }
    }
}

fn clamp_float(rect: Rect, screen: Rect) -> Rect {
    if !super::targets::usable(screen) {
        return rect;
    }
    let size = rect.size().max(vec2(1.0, 1.0)).min(screen.size());
    let min = rect.min.max(screen.min).min(screen.max - size);
    Rect::from_min_size(min, size)
}

fn split_bounds(available: f32, axis: SplitAxis, first: PanelLimits, second: PanelLimits) -> (f32, f32) {
    let (a_min, a_max, b_min, b_max) = match axis {
        SplitAxis::Horizontal => (first.min.x, first.max.x, second.min.x, second.max.x),
        SplitAxis::Vertical => (first.min.y, first.max.y, second.min.y, second.max.y),
    };
    let scale = if a_min + b_min > available { available / (a_min + b_min).max(1.0) } else { 1.0 };
    let lo = (a_min * scale).min(available);
    let hi = (available - b_min * scale).max(lo).min(available);
    let constrained_lo = lo.max(available - b_max);
    let constrained_hi = hi.min(a_max);
    if constrained_lo <= constrained_hi { (constrained_lo, constrained_hi) } else { (lo, hi) }
}

fn node_limits<P>(
    node: &Node<P>,
    style: &DockStyle,
    permissions: &impl Fn(&P) -> Permissions,
    limits_fn: &impl Fn(&P) -> PanelLimits,
) -> PanelLimits {
    match node {
        Node::Tabs { panels, .. } => {
            let mut limits = PanelLimits::default();
            for panel in panels {
                let policy = (limits_fn)(panel);
                limits.min = limits.min.max(policy.min).max(Vec2::splat(style.min_pane));
                limits.max = limits.max.min(policy.max);
            }
            limits.max = limits.max.max(limits.min);
            if panels.len() != 1 || panels.first().is_some_and(|p| (permissions)(p) != Permissions::PROTECTED) {
                limits.min.y = (limits.min.y + style.tab_height).min(1_000_000.0);
                limits.max.y = (limits.max.y + style.tab_height).min(1_000_000.0);
            }
            limits
        }
        Node::Stack { entries } => {
            let mut limits = PanelLimits::default();
            for entry in entries {
                let policy = (limits_fn)(&entry.panel);
                limits.min.x = limits.min.x.max(policy.min.x).max(style.min_pane);
                limits.min.y += if entry.open { policy.min.y.max(style.min_pane) } else { 0.0 } + style.tab_height;
                limits.max.x = limits.max.x.min(policy.max.x);
            }
            limits.min = limits.min.min(Vec2::splat(1_000_000.0));
            limits.max = limits.max.max(limits.min);
            limits
        }
        Node::Split { axis, first, second, .. } => {
            let a = node_limits(first, style, permissions, limits_fn);
            let b = node_limits(second, style, permissions, limits_fn);
            let limits = match axis {
                SplitAxis::Horizontal => PanelLimits {
                    min: vec2(a.min.x + b.min.x + style.gap, a.min.y.max(b.min.y)),
                    max: vec2(a.max.x + b.max.x + style.gap, a.max.y.min(b.max.y)),
                },
                SplitAxis::Vertical => PanelLimits {
                    min: vec2(a.min.x.max(b.min.x), a.min.y + b.min.y + style.gap),
                    max: vec2(a.max.x.min(b.max.x), a.max.y + b.max.y + style.gap),
                },
            };
            PanelLimits { min: limits.min.min(Vec2::splat(1_000_000.0)), max: limits.max.max(limits.min).min(Vec2::splat(1_000_000.0)) }
        }
    }
}

/// Measure full group rectangles with exactly the constraints used by DockArea. Validates
/// the tree first, bounding recursion and rejecting hostile geometry before traversing it.
pub fn measure_layout<P: Clone + Eq + Hash>(
    layout: &Layout<P>,
    workspace: Rect,
    screen: Rect,
    style: &DockStyle,
    permissions: impl Fn(&P) -> Permissions,
    limits: impl Fn(&P) -> PanelLimits,
) -> Result<Vec<(P, Rect)>, DockError> {
    layout.validate()?;
    if !style.valid() || !super::targets::usable(workspace) || !super::targets::usable(screen) || layout.panels().iter().any(|p| !limits(p).valid()) {
        return Err(DockError::InvalidGeometry);
    }
    let mut output = Vec::new();
    if let Some(root) = &layout.root {
        measure_node(root, workspace, style, &permissions, &limits, &mut output);
    }
    for group in &layout.floating {
        let policy = node_limits(&Node::Tabs { panels: group.panels.clone(), active: group.active }, style, &permissions, &limits);
        let [x, y, w, h] = group.rect;
        let rect = clamp_float(Rect::from_min_size(pos2(x, y), vec2(w, h).max(policy.min).min(policy.max)), screen);
        output.extend(group.panels.iter().cloned().map(|p| (p, rect)));
    }
    Ok(output)
}

fn measure_node<P: Clone>(
    node: &Node<P>,
    rect: Rect,
    style: &DockStyle,
    permissions: &impl Fn(&P) -> Permissions,
    limits: &impl Fn(&P) -> PanelLimits,
    output: &mut Vec<(P, Rect)>,
) {
    match node {
        Node::Split { axis, size, first, second } => {
            let Some((split, _, _)) = constrained_split(
                rect,
                *axis,
                *size,
                style.gap,
                node_limits(first, style, permissions, limits),
                node_limits(second, style, permissions, limits),
            ) else {
                return;
            };
            measure_node(first, split.first, style, permissions, limits, output);
            measure_node(second, split.second, style, permissions, limits, output);
        }
        Node::Tabs { panels, .. } => output.extend(panels.iter().cloned().map(|p| (p, rect))),
        Node::Stack { entries } => {
            let header = style.tab_height.min(rect.height() / entries.len().max(1) as f32);
            let remaining = (rect.height() - (entries.len() as f32 * style.tab_height).min(rect.height())).max(0.0);
            let total: f64 = entries.iter().filter(|e| e.open).map(|e| f64::from(e.height.unwrap_or(style.min_pane).max(1.0))).sum();
            let mut top = rect.top();
            for entry in entries {
                let share = if entry.open && total > 0.0 {
                    (f64::from(remaining) * f64::from(entry.height.unwrap_or(style.min_pane).max(1.0)) / total) as f32
                } else {
                    0.0
                };
                let pane = Rect::from_min_max(pos2(rect.left(), top), pos2(rect.right(), (top + header + share).min(rect.bottom())));
                output.push((entry.panel.clone(), pane));
                top = pane.bottom();
            }
        }
    }
}

fn constrained_split(
    rect: Rect,
    axis: SplitAxis,
    size: SplitSize,
    gap: f32,
    first: PanelLimits,
    second: PanelLimits,
) -> Option<(crate::layout::SplitLayout, f32, f32)> {
    let wanted = split_rect(rect, axis, size, gap, 0.0)?;
    let (lo, hi) = split_bounds(wanted.available(), axis, first, second);
    let split = split_rect(rect, axis, SplitSize::FixedFirst(wanted.first_size().clamp(lo, hi)), gap, 0.0)?;
    Some((split, lo, hi))
}
