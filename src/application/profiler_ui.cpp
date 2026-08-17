#include "profiler_ui.hpp"

#include <base/format.hpp>

#include <imgui.h>
#include <algorithm>
#include <cmath>

namespace {
    constexpr float ROW_HEIGHT = 20.0f;
    constexpr float ROW_PADDING = 2.0f;
    constexpr float OVERVIEW_HEIGHT = 60.0f;

    // Deterministic name -> color mapping, so a given zone name always gets
    // the same color across frames (makes the timeline easier to scan).
    auto color_for_name(char const *name) -> ImU32 {
        uint32_t hash = 2166136261u;
        for (char const *c = name; *c != '\0'; ++c) {
            hash = (hash ^ static_cast<uint32_t>(*c)) * 16777619u;
        }
        auto hue = static_cast<float>(hash % 360u) / 360.0f;
        return ImGui::ColorConvertFloat4ToU32(static_cast<ImVec4>(ImColor::HSV(hue, 0.55f, 0.85f)));
    }

    auto max_depth_of(ProfileTimestamp const &zone) -> int {
        int result = 0;
        for (int i = 0; i < zone.children.size; ++i) {
            result = std::max(result, 1 + max_depth_of(zone.children[i]));
        }
        return result;
    }

    void draw_zone(ImDrawList *draw_list, ImVec2 origin, float px_per_ms, ProfileTimestamp const &zone, int depth) {
        auto const name = zone.name != nullptr ? zone.name : "<unnamed>";

        float const x0 = origin.x + zone.start * px_per_ms;
        float const x1 = std::max(origin.x + zone.end * px_per_ms, x0 + 0.5f);
        float const y0 = origin.y + static_cast<float>(depth) * (ROW_HEIGHT + ROW_PADDING);
        float const y1 = y0 + ROW_HEIGHT;

        auto const color = color_for_name(name);
        draw_list->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), color, 2.0f);
        // draw_list->AddRect(ImVec2(x0, y0), ImVec2(x1, y1), IM_COL32(0, 0, 0, 128), 2.0f);

        // Draw as much of the label as fits, rather than only showing it once
        // the whole thing fits: the clip rect truncates it for us.
        if ((x1 - x0) > 6.0f) {
            auto const label = format("%s (%.3f ms)", name, static_cast<double>(zone.end - zone.start));
            draw_list->PushClipRect(ImVec2(x0, y0), ImVec2(x1, y1), true);
            draw_list->AddText(ImVec2(x0 + 2.0f, y0 + 2.0f), IM_COL32(0, 0, 0, 255), label.data);
            draw_list->PopClipRect();
        }

        if (ImGui::IsMouseHoveringRect(ImVec2(x0, y0), ImVec2(x1, y1))) {
            ImGui::SetTooltip("%s\nstart: %.4f ms\nend: %.4f ms\nduration: %.4f ms", name, static_cast<double>(zone.start), static_cast<double>(zone.end), static_cast<double>(zone.end - zone.start));
        }

        for (int i = 0; i < zone.children.size; ++i) {
            draw_zone(draw_list, origin, px_per_ms, zone.children[i], depth + 1);
        }
    }
} // namespace

ProfilerUi::ProfilerUi() {
    profiler_begin_frame();
}

void ProfilerUi::update(GpuContext &gpu_context) {
    profiler_end_frame();

    if (!paused) {
        auto const slot_index = frame_count % static_cast<uint64_t>(FRAME_HISTORY_COUNT);
        auto const thread_count = std::min<uint64_t>(profiler_get_thread_count(), static_cast<uint64_t>(MAX_DISPLAYED_THREADS));
        frame_thread_counts[slot_index] = thread_count;

        float duration = 0.0f;
        for (uint64_t t = 0; t < thread_count; ++t) {
            auto &slot = frames[slot_index][t];
            if (t == MAX_DISPLAYED_THREADS - 1)
                gpu_context.get_timestamps(frames[slot_index][t]);
            else
                profiler_resolve_frame(slot, t);
            for (int i = 0; i < slot.size; ++i) {
                duration = std::max(duration, slot[i].end);
            }
        }
        // Clear any lanes that were in use in a previous occupant of this ring
        // buffer slot but aren't active anymore, so stale data doesn't linger.
        for (uint64_t t = thread_count; t < static_cast<uint64_t>(MAX_DISPLAYED_THREADS); ++t) {
            frames[slot_index][t].clear();
        }
        frame_durations[slot_index] = duration;

        ++frame_count;
    } else {
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(10ms);
    }

    profiler_begin_frame();
}

void ProfilerUi::ui_fullscreen() {
    ImGuiViewport const *viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGuiWindowFlags const flags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
        ImGuiWindowFlags_NoSavedSettings;
    ImGui::Begin("##profiler_fullscreen", nullptr, flags);
    ImGui::TextUnformatted("Profiler view (F7 to exit)");
    ImGui::Separator();
    draw_contents();
    ImGui::End();
}

void ProfilerUi::ui_timeline() {
    auto const history_size = std::min<uint64_t>(frame_count, static_cast<uint64_t>(FRAME_HISTORY_COUNT));
    auto const oldest_frame = frame_count - history_size;

    // Frame-time overview strip: one bar per recorded frame, click a bar to
    // pin the flame-graph below to that frame.
    auto const overview_size = ImVec2(ImGui::GetContentRegionAvail().x, OVERVIEW_HEIGHT);
    ImGui::InvisibleButton("##profiler_overview", overview_size);
    auto const overview_min = ImGui::GetItemRectMin();
    auto const overview_max = ImGui::GetItemRectMax();
    auto *overview_draw_list = ImGui::GetWindowDrawList();
    overview_draw_list->AddRectFilled(overview_min, overview_max, IM_COL32(20, 20, 20, 255));

    float max_duration = 1.0f;
    for (uint64_t i = 0; i < history_size; ++i) {
        max_duration = std::max(max_duration, frame_durations[(oldest_frame + i) % static_cast<uint64_t>(FRAME_HISTORY_COUNT)]);
    }

    auto const bar_width = overview_size.x / static_cast<float>(history_size);
    for (uint64_t i = 0; i < history_size; ++i) {
        auto const frame_index = oldest_frame + i;
        auto const duration = frame_durations[frame_index % static_cast<uint64_t>(FRAME_HISTORY_COUNT)];
        auto const height_frac = std::clamp(duration / max_duration, 0.0f, 1.0f);
        auto const x0 = overview_min.x + static_cast<float>(i) * bar_width;
        auto const x1 = x0 + std::max(bar_width - 1.0f, 1.0f);
        auto const y1 = overview_max.y;
        auto const y0 = y1 - height_frac * overview_size.y;

        auto color = duration > (1000.0f / 60.0f) ? IM_COL32(230, 80, 80, 255) : IM_COL32(100, 200, 100, 255);
        overview_draw_list->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), color);
    }
}

void ProfilerUi::draw_contents() {
    if (frame_count == 0) {
        ImGui::TextUnformatted("No frames recorded yet.");
        return;
    }

    auto const history_size = std::min<uint64_t>(frame_count, static_cast<uint64_t>(FRAME_HISTORY_COUNT));
    auto const oldest_frame = frame_count - history_size;
    auto const latest_frame = frame_count - 1;
    auto display_frame = selected_frame < 0 ? latest_frame : static_cast<uint64_t>(selected_frame);
    display_frame = std::clamp(display_frame, oldest_frame, latest_frame);

    // Left/Right arrows step through frame history, as long as nothing else
    // (e.g. a text field) wants the keyboard.
    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) && !ImGui::GetIO().WantTextInput) {
        if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow) && display_frame > oldest_frame) {
            selected_frame = static_cast<int64_t>(display_frame - 1);
            display_frame -= 1;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_RightArrow)) {
            selected_frame = (display_frame + 1 >= latest_frame) ? -1 : static_cast<int64_t>(display_frame + 1);
            display_frame = selected_frame < 0 ? latest_frame : static_cast<uint64_t>(selected_frame);
        }
    }

    ImGui::Text("%llu frames recorded (showing last %llu)", static_cast<unsigned long long>(frame_count), static_cast<unsigned long long>(history_size));

    // Frame-time overview strip: one bar per recorded frame, click a bar to
    // pin the flame-graph below to that frame.
    auto const overview_size = ImVec2(ImGui::GetContentRegionAvail().x, OVERVIEW_HEIGHT);
    ImGui::InvisibleButton("##profiler_overview", overview_size);
    auto const overview_min = ImGui::GetItemRectMin();
    auto const overview_max = ImGui::GetItemRectMax();
    auto *overview_draw_list = ImGui::GetWindowDrawList();
    overview_draw_list->AddRectFilled(overview_min, overview_max, IM_COL32(20, 20, 20, 255));

    float max_duration = 1.0f;
    for (uint64_t i = 0; i < history_size; ++i) {
        max_duration = std::max(max_duration, frame_durations[(oldest_frame + i) % static_cast<uint64_t>(FRAME_HISTORY_COUNT)]);
    }

    auto const bar_width = overview_size.x / static_cast<float>(history_size);
    for (uint64_t i = 0; i < history_size; ++i) {
        auto const frame_index = oldest_frame + i;
        auto const duration = frame_durations[frame_index % static_cast<uint64_t>(FRAME_HISTORY_COUNT)];
        auto const height_frac = std::clamp(duration / max_duration, 0.0f, 1.0f);
        auto const x0 = overview_min.x + static_cast<float>(i) * bar_width;
        auto const x1 = x0 + std::max(bar_width - 1.0f, 1.0f);
        auto const y1 = overview_max.y;
        auto const y0 = y1 - height_frac * overview_size.y;

        bool const is_shown = frame_index == display_frame;
        auto color = duration > (1000.0f / 60.0f) ? IM_COL32(230, 80, 80, 255) : IM_COL32(100, 200, 100, 255);
        if (is_shown) {
            color = IM_COL32(255, 210, 60, 255);
        }
        overview_draw_list->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), color);
    }

    if (ImGui::IsItemHovered()) {
        auto const mouse_x = ImGui::GetIO().MousePos.x - overview_min.x;
        auto const hovered_i = static_cast<uint64_t>(std::clamp(mouse_x / bar_width, 0.0f, static_cast<float>(history_size - 1)));
        auto const hovered_frame = oldest_frame + hovered_i;
        ImGui::SetTooltip("frame %llu: %.3f ms", static_cast<unsigned long long>(hovered_frame), static_cast<double>(frame_durations[hovered_frame % static_cast<uint64_t>(FRAME_HISTORY_COUNT)]));
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            selected_frame = static_cast<int64_t>(hovered_frame);
            display_frame = hovered_frame;
        }
    }

    // Buttons get their own row: the overview strip above is already full
    // width, so a SameLine() right after it pushes them off-screen.
    if (ImGui::Button("< Prev") && display_frame > oldest_frame) {
        selected_frame = static_cast<int64_t>(display_frame - 1);
    }
    ImGui::SameLine();
    if (ImGui::Button("Next >")) {
        selected_frame = (display_frame + 1 >= latest_frame) ? -1 : static_cast<int64_t>(display_frame + 1);
    }
    ImGui::SameLine();
    if (ImGui::Button("Follow Latest")) {
        selected_frame = -1;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(left/right arrows step frames, mouse wheel zooms, click+drag pans)");

    ImGui::Separator();

    display_frame = selected_frame < 0 ? latest_frame : static_cast<uint64_t>(selected_frame);
    display_frame = std::clamp(display_frame, oldest_frame, latest_frame);

    auto const slot_index = display_frame % static_cast<uint64_t>(FRAME_HISTORY_COUNT);
    auto const thread_count = frame_thread_counts[slot_index];
    auto const frame_duration = std::max(frame_durations[slot_index], 0.001f);

    ImGui::Text("Frame %llu - %.3f ms - %llu thread%s", static_cast<unsigned long long>(display_frame), static_cast<double>(frame_duration), static_cast<unsigned long long>(thread_count), thread_count == 1 ? "" : "s");

    // Height of each thread's lane (just its call-stack depth: the thread ID
    // is drawn in a gutter to the left instead of a row above), stacked
    // vertically so every thread gets its own flame graph.
    constexpr float THREAD_GAP = 8.0f;
    int thread_depths[MAX_DISPLAYED_THREADS] = {};
    float total_height = 0.0f;
    for (uint64_t t = 0; t < thread_count; ++t) {
        auto const &zones = frames[slot_index][t];
        int depth = 0;
        for (int i = 0; i < zones.size; ++i) {
            depth = std::max(depth, max_depth_of(zones[i]));
        }
        thread_depths[t] = depth;
        total_height += static_cast<float>(depth + 1) * (ROW_HEIGHT + ROW_PADDING) + THREAD_GAP;
    }

    auto const avail = ImGui::GetContentRegionAvail();
    auto const base_px_per_ms = avail.x / frame_duration;
    auto px_per_ms = base_px_per_ms * zoom;

    // No scrollbars: the mouse wheel zooms instead of scrolling, and panning
    // is done by click+drag (below) rather than dragging a scrollbar.
    ImGui::BeginChild("##profiler_flamegraph", ImVec2(0.0f, std::max(avail.y, 100.0f)), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    auto const &io = ImGui::GetIO();
    bool const hovered = ImGui::IsWindowHovered();

    // ImGui::SetScrollX/Y() only sets a *target*: the actual Scroll value (and
    // therefore GetCursorScreenPos()) doesn't catch up until this window's
    // next Begin(), one frame later. If we changed the scroll below and then
    // asked ImGui for the cursor position, we'd draw this frame's (new) zoom
    // against last frame's (stale) scroll and get a one-frame misaligned
    // flash. So: anchor content-space (0,0) to screen space now, before any
    // scroll change, and derive this frame's origin from that anchor plus
    // whatever scroll value we decide on ourselves - no dependency on ImGui
    // having applied the target yet.
    auto const content_zero = ImVec2(ImGui::GetCursorScreenPos().x + ImGui::GetScrollX(), ImGui::GetCursorScreenPos().y + ImGui::GetScrollY());
    auto scroll_x = ImGui::GetScrollX();
    auto scroll_y = ImGui::GetScrollY();

    // Mouse-wheel zoom, keeping the ms position under the cursor stationary.
    if (hovered && io.MouseWheel != 0.0f) {
        auto const window_x = ImGui::GetWindowPos().x;
        auto const mouse_ms = (io.MousePos.x - window_x + scroll_x) / px_per_ms;

        auto const zoom_factor = std::pow(1.2f, io.MouseWheel);
        zoom = std::clamp(zoom * zoom_factor, 1.0f, 2000.0f);
        px_per_ms = base_px_per_ms * zoom;

        scroll_x = std::max(mouse_ms * px_per_ms - (io.MousePos.x - window_x), 0.0f);
        ImGui::SetScrollX(scroll_x);
    }

    // Click+drag panning. Once started, keeps panning even if the cursor
    // strays outside the child window, and stops as soon as the button is
    // released (wherever that happens to be).
    if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        is_panning_flamegraph = true;
    }
    if (is_panning_flamegraph) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            scroll_x = std::max(scroll_x - io.MouseDelta.x, 0.0f);
            scroll_y = std::max(scroll_y - io.MouseDelta.y, 0.0f);
            ImGui::SetScrollX(scroll_x);
            ImGui::SetScrollY(scroll_y);
        } else {
            is_panning_flamegraph = false;
        }
    }

    auto const canvas_width = frame_duration * px_per_ms;
    auto const canvas_height = std::max(total_height, 1.0f);
    auto const origin = ImVec2(content_zero.x - scroll_x, content_zero.y - scroll_y);
    auto const window_pos = ImGui::GetWindowPos();
    ImGui::Dummy(ImVec2(canvas_width, canvas_height));
    auto *flamegraph_draw_list = ImGui::GetWindowDrawList();

    float y_offset = 0.0f;
    for (uint64_t t = 0; t < thread_count; ++t) {
        auto const &zones = frames[slot_index][t];
        auto const thread_origin = ImVec2(origin.x, origin.y + y_offset);
        for (int i = 0; i < zones.size; ++i) {
            draw_zone(flamegraph_draw_list, thread_origin, px_per_ms, zones[i], 0);
        }

        // Thread ID label, pinned to the left edge of the visible area (not
        // the scrolled content) and drawn on top so it stays legible as bars
        // pan underneath it.
        auto const label = t == MAX_DISPLAYED_THREADS - 1 ? format("GPU") : format("Thread %llu", static_cast<unsigned long long>(t));
        auto const label_size = ImGui::CalcTextSize(label.data);
        auto const label_pos = ImVec2(window_pos.x + 4.0f, thread_origin.y + 2.0f);
        flamegraph_draw_list->AddRectFilled(
            ImVec2(label_pos.x - 2.0f, label_pos.y - 1.0f),
            ImVec2(label_pos.x + label_size.x + 2.0f, label_pos.y + label_size.y + 1.0f),
            IM_COL32(0, 0, 0, 180), 2.0f);
        flamegraph_draw_list->AddText(label_pos, IM_COL32(255, 255, 255, 255), label.data);

        y_offset += static_cast<float>(thread_depths[t] + 1) * (ROW_HEIGHT + ROW_PADDING) + THREAD_GAP;
    }

    ImGui::EndChild();
}
