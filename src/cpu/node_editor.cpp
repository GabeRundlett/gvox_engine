#include "node_editor.hpp"

#include <imgui.h>

#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_internal.h>

enum class IconType : ImU32 {
    Flow,
    Circle,
    Square,
    Grid,
    RoundSquare,
    Diamond,
};

static inline ImRect ImGui_GetItemRect() {
    return ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
}

static inline ImRect ImRect_Expanded(const ImRect &rect, float x, float y) {
    auto result = rect;
    result.Min.x -= x;
    result.Min.y -= y;
    result.Max.x += x;
    result.Max.y += y;
    return result;
}

static void DrawIcon(ImDrawList *drawList, const ImVec2 &a, const ImVec2 &b, IconType type, bool filled, ImU32 color, ImU32 innerColor) {
    auto rect = ImRect(a, b);
    // auto rect_x = rect.Min.x;
    auto rect_y = rect.Min.y;
    auto rect_w = rect.Max.x - rect.Min.x;
    auto rect_h = rect.Max.y - rect.Min.y;
    auto rect_center_x = (rect.Min.x + rect.Max.x) * 0.5f;
    auto rect_center_y = (rect.Min.y + rect.Max.y) * 0.5f;
    auto rect_center = ImVec2(rect_center_x, rect_center_y);
    const auto outline_scale = rect_w / 24.0f;
    const auto extra_segments = static_cast<int>(2 * outline_scale); // for full circle

    if (type == IconType::Flow) {
        const auto origin_scale = rect_w / 24.0f;

        const auto offset_x = 1.0f * origin_scale;
        const auto offset_y = 0.0f * origin_scale;
        const auto margin = (filled ? 2.0f : 2.0f) * origin_scale;
        const auto rounding = 0.1f * origin_scale;
        const auto tip_round = 0.7f; // percentage of triangle edge (for tip)
        // const auto edge_round = 0.7f; // percentage of triangle edge (for corner)
        const auto canvas = ImRect(
            rect.Min.x + margin + offset_x,
            rect.Min.y + margin + offset_y,
            rect.Max.x - margin + offset_x,
            rect.Max.y - margin + offset_y);
        const auto canvas_x = canvas.Min.x;
        const auto canvas_y = canvas.Min.y;
        const auto canvas_w = canvas.Max.x - canvas.Min.x;
        const auto canvas_h = canvas.Max.y - canvas.Min.y;

        const auto left = canvas_x + canvas_w * 0.5f * 0.3f;
        const auto right = canvas_x + canvas_w - canvas_w * 0.5f * 0.3f;
        const auto top = canvas_y + canvas_h * 0.5f * 0.2f;
        const auto bottom = canvas_y + canvas_h - canvas_h * 0.5f * 0.2f;
        const auto center_y = (top + bottom) * 0.5f;
        // const auto angle = AX_PI * 0.5f * 0.5f * 0.5f;

        const auto tip_top = ImVec2(canvas_x + canvas_w * 0.5f, top);
        const auto tip_right = ImVec2(right, center_y);
        const auto tip_bottom = ImVec2(canvas_x + canvas_w * 0.5f, bottom);

        drawList->PathLineTo(ImVec2(left, top) + ImVec2(0, rounding));
        drawList->PathBezierCubicCurveTo(
            ImVec2(left, top),
            ImVec2(left, top),
            ImVec2(left, top) + ImVec2(rounding, 0));
        drawList->PathLineTo(tip_top);
        drawList->PathLineTo(tip_top + (tip_right - tip_top) * tip_round);
        drawList->PathBezierCubicCurveTo(
            tip_right,
            tip_right,
            tip_bottom + (tip_right - tip_bottom) * tip_round);
        drawList->PathLineTo(tip_bottom);
        drawList->PathLineTo(ImVec2(left, bottom) + ImVec2(rounding, 0));
        drawList->PathBezierCubicCurveTo(
            ImVec2(left, bottom),
            ImVec2(left, bottom),
            ImVec2(left, bottom) - ImVec2(0, rounding));

        if (!filled) {
            if (innerColor & 0xFF000000)
                drawList->AddConvexPolyFilled(drawList->_Path.Data, drawList->_Path.Size, innerColor);

            drawList->PathStroke(color, true, 2.0f * outline_scale);
        } else
            drawList->PathFillConvex(color);
    } else {
        auto triangleStart = rect_center_x + 0.32f * rect_w;

        auto rect_offset = -static_cast<int>(rect_w * 0.25f * 0.25f);

        rect.Min.x += static_cast<float>(rect_offset);
        rect.Max.x += static_cast<float>(rect_offset);
        // rect_x += static_cast<float>(rect_offset);
        rect_center_x += static_cast<float>(rect_offset) * 0.5f;
        rect_center.x += static_cast<float>(rect_offset) * 0.5f;

        if (type == IconType::Circle) {
            const auto c = rect_center;

            if (!filled) {
                const auto r = 0.5f * rect_w / 2.0f - 0.5f;

                if (innerColor & 0xFF000000)
                    drawList->AddCircleFilled(c, r, innerColor, 12 + extra_segments);
                drawList->AddCircle(c, r, color, 12 + extra_segments, 2.0f * outline_scale);
            } else {
                drawList->AddCircleFilled(c, 0.5f * rect_w / 2.0f, color, 12 + extra_segments);
            }
        }

        if (type == IconType::Square) {
            if (filled) {
                const auto r = 0.5f * rect_w / 2.0f;
                const auto p0 = rect_center - ImVec2(r, r);
                const auto p1 = rect_center + ImVec2(r, r);

#if IMGUI_VERSION_NUM > 18101
                drawList->AddRectFilled(p0, p1, color, 0, ImDrawFlags_RoundCornersAll);
#else
                drawList->AddRectFilled(p0, p1, color, 0, 15);
#endif
            } else {
                const auto r = 0.5f * rect_w / 2.0f - 0.5f;
                const auto p0 = rect_center - ImVec2(r, r);
                const auto p1 = rect_center + ImVec2(r, r);

                if (innerColor & 0xFF000000) {
#if IMGUI_VERSION_NUM > 18101
                    drawList->AddRectFilled(p0, p1, innerColor, 0, ImDrawFlags_RoundCornersAll);
#else
                    drawList->AddRectFilled(p0, p1, innerColor, 0, 15);
#endif
                }

#if IMGUI_VERSION_NUM > 18101
                drawList->AddRect(p0, p1, color, 0, ImDrawFlags_RoundCornersAll, 2.0f * outline_scale);
#else
                drawList->AddRect(p0, p1, color, 0, 15, 2.0f * outline_scale);
#endif
            }
        }

        if (type == IconType::Grid) {
            const auto r = 0.5f * rect_w / 2.0f;
            const auto w = ceilf(r / 3.0f);

            const auto baseTl = ImVec2(floorf(rect_center_x - w * 2.5f), floorf(rect_center_y - w * 2.5f));
            const auto baseBr = ImVec2(floorf(baseTl.x + w), floorf(baseTl.y + w));

            auto tl = baseTl;
            auto br = baseBr;
            for (int i = 0; i < 3; ++i) {
                tl.x = baseTl.x;
                br.x = baseBr.x;
                drawList->AddRectFilled(tl, br, color);
                tl.x += w * 2;
                br.x += w * 2;
                if (i != 1 || filled)
                    drawList->AddRectFilled(tl, br, color);
                tl.x += w * 2;
                br.x += w * 2;
                drawList->AddRectFilled(tl, br, color);

                tl.y += w * 2;
                br.y += w * 2;
            }

            triangleStart = br.x + w + 1.0f / 24.0f * rect_w;
        }

        if (type == IconType::RoundSquare) {
            if (filled) {
                const auto r = 0.5f * rect_w / 2.0f;
                const auto cr = r * 0.5f;
                const auto p0 = rect_center - ImVec2(r, r);
                const auto p1 = rect_center + ImVec2(r, r);

#if IMGUI_VERSION_NUM > 18101
                drawList->AddRectFilled(p0, p1, color, cr, ImDrawFlags_RoundCornersAll);
#else
                drawList->AddRectFilled(p0, p1, color, cr, 15);
#endif
            } else {
                const auto r = 0.5f * rect_w / 2.0f - 0.5f;
                const auto cr = r * 0.5f;
                const auto p0 = rect_center - ImVec2(r, r);
                const auto p1 = rect_center + ImVec2(r, r);

                if (innerColor & 0xFF000000) {
#if IMGUI_VERSION_NUM > 18101
                    drawList->AddRectFilled(p0, p1, innerColor, cr, ImDrawFlags_RoundCornersAll);
#else
                    drawList->AddRectFilled(p0, p1, innerColor, cr, 15);
#endif
                }

#if IMGUI_VERSION_NUM > 18101
                drawList->AddRect(p0, p1, color, cr, ImDrawFlags_RoundCornersAll, 2.0f * outline_scale);
#else
                drawList->AddRect(p0, p1, color, cr, 15, 2.0f * outline_scale);
#endif
            }
        } else if (type == IconType::Diamond) {
            if (filled) {
                const auto r = 0.607f * rect_w / 2.0f;
                const auto c = rect_center;

                drawList->PathLineTo(c + ImVec2(0, -r));
                drawList->PathLineTo(c + ImVec2(r, 0));
                drawList->PathLineTo(c + ImVec2(0, r));
                drawList->PathLineTo(c + ImVec2(-r, 0));
                drawList->PathFillConvex(color);
            } else {
                const auto r = 0.607f * rect_w / 2.0f - 0.5f;
                const auto c = rect_center;

                drawList->PathLineTo(c + ImVec2(0, -r));
                drawList->PathLineTo(c + ImVec2(r, 0));
                drawList->PathLineTo(c + ImVec2(0, r));
                drawList->PathLineTo(c + ImVec2(-r, 0));

                if (innerColor & 0xFF000000)
                    drawList->AddConvexPolyFilled(drawList->_Path.Data, drawList->_Path.Size, innerColor);

                drawList->PathStroke(color, true, 2.0f * outline_scale);
            }
        } else {
            const auto triangleTip = triangleStart + rect_w * (0.45f - 0.32f);

            drawList->AddTriangleFilled(
                ImVec2(ceilf(triangleTip), rect_y + rect_h * 0.5f),
                ImVec2(triangleStart, rect_center_y + 0.15f * rect_h),
                ImVec2(triangleStart, rect_center_y - 0.15f * rect_h),
                color);
        }
    }
}

static void Icon(const ImVec2 &size, IconType type, bool filled, const ImVec4 &color /* = ImVec4(1, 1, 1, 1)*/, const ImVec4 &innerColor /* = ImVec4(0, 0, 0, 0)*/) {
    if (ImGui::IsRectVisible(size)) {
        auto cursorPos = ImGui::GetCursorScreenPos();
        auto drawList = ImGui::GetWindowDrawList();
        DrawIcon(drawList, cursorPos, cursorPos + size, type, filled, ImColor(color), ImColor(innerColor));
    }

    ImGui::Dummy(size);
}

static bool CanCreateLink(Pin *a, Pin *b) {
    if (!a || !b || a == b || a->Kind == b->Kind || a->Type != b->Type || a->Node == b->Node)
        return false;

    return true;
}

static ImColor GetIconColor(PinType type) {
    switch (type) {
    default:
    case PinType::Flow: return ImColor(255, 255, 255);
    case PinType::Bool: return ImColor(220, 48, 48);
    case PinType::Int: return ImColor(68, 201, 156);
    case PinType::Float: return ImColor(147, 226, 74);
    case PinType::String: return ImColor(124, 21, 153);
    case PinType::Object: return ImColor(51, 150, 215);
    case PinType::Function: return ImColor(218, 0, 183);
    case PinType::Delegate: return ImColor(255, 48, 48);
    }
};

static void DrawPinIcon(const Pin &pin, bool connected, int alpha, float m_PinIconSize) {
    IconType iconType;
    ImColor color = GetIconColor(pin.Type);
    color.Value.w = static_cast<float>(alpha) / 255.0f;
    switch (pin.Type) {
    case PinType::Flow: iconType = IconType::Flow; break;
    case PinType::Bool: iconType = IconType::Circle; break;
    case PinType::Int: iconType = IconType::Circle; break;
    case PinType::Float: iconType = IconType::Circle; break;
    case PinType::String: iconType = IconType::Circle; break;
    case PinType::Object: iconType = IconType::Circle; break;
    case PinType::Function: iconType = IconType::Circle; break;
    case PinType::Delegate: iconType = IconType::Square; break;
    default:
        return;
    }

    Icon(ImVec2(m_PinIconSize, m_PinIconSize), iconType, connected, color, ImColor(32, 32, 32, alpha));
};

struct BlueprintNodeBuilder {
    BlueprintNodeBuilder(ImTextureID texture = nullptr, int textureWidth = 0, int textureHeight = 0);

    void Begin(imgui_node_editor::NodeId id);
    void End();

    void Header(const ImVec4 &color = ImVec4(1, 1, 1, 1));
    void EndHeader();

    void Input(imgui_node_editor::PinId id);
    void EndInput();

    void Middle();

    void Output(imgui_node_editor::PinId id);
    void EndOutput();

  private:
    enum class Stage {
        Invalid,
        Begin,
        Header,
        Content,
        Input,
        Output,
        Middle,
        End
    };

    bool SetStage(Stage stage);

    void Pin(imgui_node_editor::PinId id, ax::NodeEditor::PinKind kind);
    void EndPin();

    ImTextureID HeaderTextureId;
    int HeaderTextureWidth;
    int HeaderTextureHeight;
    imgui_node_editor::NodeId CurrentNodeId;
    Stage CurrentStage;
    ImU32 HeaderColor;
    ImVec2 NodeMin;
    ImVec2 NodeMax;
    ImVec2 HeaderMin;
    ImVec2 HeaderMax;
    ImVec2 ContentMin;
    ImVec2 ContentMax;
    bool HasHeader;
};

BlueprintNodeBuilder::BlueprintNodeBuilder(ImTextureID texture, int textureWidth, int textureHeight)
    : HeaderTextureId(texture),
      HeaderTextureWidth(textureWidth),
      HeaderTextureHeight(textureHeight),
      CurrentNodeId(0),
      CurrentStage(Stage::Invalid),
      HasHeader(false) {
}

void BlueprintNodeBuilder::Begin(imgui_node_editor::NodeId id) {
    HasHeader = false;
    HeaderMin = HeaderMax = ImVec2();
    imgui_node_editor::PushStyleVar(imgui_node_editor::StyleVar_NodePadding, ImVec4(8, 4, 8, 8));
    imgui_node_editor::BeginNode(id);
    ImGui::PushID(id.AsPointer());
    CurrentNodeId = id;
    SetStage(Stage::Begin);
}

void BlueprintNodeBuilder::End() {
    SetStage(Stage::End);
    imgui_node_editor::EndNode();
    if (ImGui::IsItemVisible()) {
        auto alpha = static_cast<int>(255 * ImGui::GetStyle().Alpha);
        auto drawList = imgui_node_editor::GetNodeBackgroundDrawList(CurrentNodeId);
        const auto halfBorderWidth = imgui_node_editor::GetStyle().NodeBorderWidth * 0.5f;
        auto headerColor = IM_COL32(0, 0, 0, alpha) | (HeaderColor & IM_COL32(255, 255, 255, 0));
        if ((HeaderMax.x > HeaderMin.x) && (HeaderMax.y > HeaderMin.y) && HeaderTextureId) {
            const auto uv = ImVec2(
                (HeaderMax.x - HeaderMin.x) / 4.0f * static_cast<float>(HeaderTextureWidth),
                (HeaderMax.y - HeaderMin.y) / 4.0f * static_cast<float>(HeaderTextureHeight));
            drawList->AddImageRounded(
                HeaderTextureId,
                ImVec2(HeaderMin.x - 8 - halfBorderWidth, HeaderMin.y - 4 - halfBorderWidth),
                ImVec2(HeaderMax.x + 8 - halfBorderWidth, HeaderMax.y + 0),
                ImVec2(0.0f, 0.0f), uv,
                headerColor, imgui_node_editor::GetStyle().NodeRounding, ImDrawFlags_RoundCornersTop);
            auto headerSeparatorMin = ImVec2(HeaderMin.x, HeaderMax.y);
            auto headerSeparatorMax = ImVec2(HeaderMax.x, HeaderMin.y);
            if ((headerSeparatorMax.x > headerSeparatorMin.x) && (headerSeparatorMax.y > headerSeparatorMin.y)) {
                drawList->AddLine(
                    ImVec2(headerSeparatorMin.x + -(8 - halfBorderWidth), headerSeparatorMin.y + -0.5f),
                    ImVec2(headerSeparatorMax.x + (8 - halfBorderWidth), headerSeparatorMax.y + -0.5f),
                    ImColor(255, 255, 255, 96 * alpha / (3 * 255)), 1.0f);
            }
        }
    }
    CurrentNodeId = 0;
    ImGui::PopID();
    imgui_node_editor::PopStyleVar();
    SetStage(Stage::Invalid);
}

void BlueprintNodeBuilder::Header(const ImVec4 &color) {
    HeaderColor = ImColor(color);
    SetStage(Stage::Header);
}

void BlueprintNodeBuilder::EndHeader() {
    SetStage(Stage::Content);
}

void BlueprintNodeBuilder::Input(imgui_node_editor::PinId id) {
    if (CurrentStage == Stage::Begin)
        SetStage(Stage::Content);
    SetStage(Stage::Input);
    // const auto applyPadding = (CurrentStage == Stage::Input);
    // if (applyPadding)
    //     ImGui::Spring(0);
    Pin(id, imgui_node_editor::PinKind::Input);
    // ImGui::BeginHorizontal(id.AsPointer());
}

void BlueprintNodeBuilder::EndInput() {
    // ImGui::EndHorizontal();
    EndPin();
}

void BlueprintNodeBuilder::Middle() {
    if (CurrentStage == Stage::Begin)
        SetStage(Stage::Content);
    SetStage(Stage::Middle);
}

void BlueprintNodeBuilder::Output(imgui_node_editor::PinId id) {
    if (CurrentStage == Stage::Begin)
        SetStage(Stage::Content);
    SetStage(Stage::Output);
    // const auto applyPadding = (CurrentStage == Stage::Output);
    // if (applyPadding)
    //     ImGui::Spring(0);
    Pin(id, imgui_node_editor::PinKind::Output);
    // ImGui::BeginHorizontal(id.AsPointer());
}

void BlueprintNodeBuilder::EndOutput() {
    // ImGui::EndHorizontal();
    EndPin();
}

bool BlueprintNodeBuilder::SetStage(Stage stage) {
    if (stage == CurrentStage)
        return false;
    auto oldStage = CurrentStage;
    CurrentStage = stage;
    // ImVec2 cursor;
    switch (oldStage) {
    case Stage::Begin:
        break;
    case Stage::Header:
        // ImGui::EndHorizontal();
        HeaderMin = ImGui::GetItemRectMin();
        HeaderMax = ImGui::GetItemRectMax();
        // spacing between header and content
        // ImGui::Spring(0, ImGui::GetStyle().ItemSpacing.y * 2.0f);
        break;
    case Stage::Content:
        break;
    case Stage::Input:
        imgui_node_editor::PopStyleVar(2);
        // ImGui::Spring(1, 0);
        // ImGui::EndVertical();
        // #debug
        // ImGui::GetWindowDrawList()->AddRect(
        //     ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), IM_COL32(255, 0, 0, 255));
        break;
    case Stage::Middle:
        // ImGui::EndVertical();
        // #debug
        // ImGui::GetWindowDrawList()->AddRect(
        //     ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), IM_COL32(255, 0, 0, 255));
        break;
    case Stage::Output:
        imgui_node_editor::PopStyleVar(2);
        // ImGui::Spring(1, 0);
        // ImGui::EndVertical();
        // #debug
        // ImGui::GetWindowDrawList()->AddRect(
        //     ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), IM_COL32(255, 0, 0, 255));
        break;
    case Stage::End:
        break;
    case Stage::Invalid:
        break;
    }
    switch (stage) {
    case Stage::Begin:
        // ImGui::BeginVertical("node");
        break;
    case Stage::Header:
        HasHeader = true;
        // ImGui::BeginHorizontal("header");
        break;
    case Stage::Content:
        // if (oldStage == Stage::Begin)
        //     ImGui::Spring(0);
        // ImGui::BeginHorizontal("content");
        // ImGui::Spring(0, 0);
        break;
    case Stage::Input:
        // ImGui::BeginVertical("inputs", ImVec2(0, 0), 0.0f);
        imgui_node_editor::PushStyleVar(imgui_node_editor::StyleVar_PivotAlignment, ImVec2(0, 0.5f));
        imgui_node_editor::PushStyleVar(imgui_node_editor::StyleVar_PivotSize, ImVec2(0, 0));
        // if (!HasHeader)
        //     ImGui::Spring(1, 0);
        break;
    case Stage::Middle:
        // ImGui::Spring(1);
        // ImGui::BeginVertical("middle", ImVec2(0, 0), 1.0f);
        break;
    case Stage::Output:
        // if (oldStage == Stage::Middle || oldStage == Stage::Input)
        //     ImGui::Spring(1);
        // else
        //     ImGui::Spring(1, 0);
        // ImGui::BeginVertical("outputs", ImVec2(0, 0), 1.0f);
        imgui_node_editor::PushStyleVar(imgui_node_editor::StyleVar_PivotAlignment, ImVec2(1.0f, 0.5f));
        imgui_node_editor::PushStyleVar(imgui_node_editor::StyleVar_PivotSize, ImVec2(0, 0));
        // if (!HasHeader)
        //     ImGui::Spring(1, 0);
        break;
    case Stage::End:
        // if (oldStage == Stage::Input)
        //     ImGui::Spring(1, 0);
        // if (oldStage != Stage::Begin)
        //     ImGui::EndHorizontal();
        ContentMin = ImGui::GetItemRectMin();
        ContentMax = ImGui::GetItemRectMax();
        // ImGui::Spring(0);
        //  ImGui::EndVertical();
        NodeMin = ImGui::GetItemRectMin();
        NodeMax = ImGui::GetItemRectMax();
        break;
    case Stage::Invalid:
        break;
    }

    return true;
}

void BlueprintNodeBuilder::Pin(imgui_node_editor::PinId id, imgui_node_editor::PinKind kind) {
    imgui_node_editor::BeginPin(id, kind);
}

void BlueprintNodeBuilder::EndPin() {
    imgui_node_editor::EndPin();
    // #debug
    // ImGui::GetWindowDrawList()->AddRectFilled(
    //     ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), IM_COL32(255, 0, 0, 64));
}

int NodeEditor::GetNextId() {
    return m_NextId++;
}

imgui_node_editor::LinkId NodeEditor::GetNextLinkId() {
    return imgui_node_editor::LinkId(static_cast<uint32_t>(GetNextId()));
}

Node *NodeEditor::FindNode(imgui_node_editor::NodeId id) {
    for (auto &node : m_Nodes)
        if (node.ID == id)
            return &node;

    return nullptr;
}

Link *NodeEditor::FindLink(imgui_node_editor::LinkId id) {
    for (auto &link : m_Links)
        if (link.ID == id)
            return &link;

    return nullptr;
}

Pin *NodeEditor::FindPin(imgui_node_editor::PinId id) {
    if (!id)
        return nullptr;

    for (auto &node : m_Nodes) {
        for (auto &pin : node.Inputs)
            if (pin.ID == id)
                return &pin;

        for (auto &pin : node.Outputs)
            if (pin.ID == id)
                return &pin;
    }

    return nullptr;
}

bool NodeEditor::IsPinLinked(imgui_node_editor::PinId id) {
    if (!id)
        return false;

    for (auto &link : m_Links)
        if (link.StartPinID == id || link.EndPinID == id)
            return true;

    return false;
}

void NodeEditor::BuildNode(Node *node) {
    for (auto &input : node->Inputs) {
        input.Node = node;
        input.Kind = PinKind::Input;
    }

    for (auto &output : node->Outputs) {
        output.Node = node;
        output.Kind = PinKind::Output;
    }
}

void NodeEditor::BuildNodes() {
    for (auto &node : m_Nodes)
        BuildNode(&node);
}

Node *NodeEditor::SpawnInputActionNode() {
    m_Nodes.emplace_back(GetNextId(), "InputAction Fire", ImColor(255, 128, 128));
    m_Nodes.back().Outputs.emplace_back(GetNextId(), "", PinType::Delegate);
    m_Nodes.back().Outputs.emplace_back(GetNextId(), "Pressed", PinType::Flow);
    m_Nodes.back().Outputs.emplace_back(GetNextId(), "Released", PinType::Flow);

    BuildNode(&m_Nodes.back());

    return &m_Nodes.back();
}

Node *NodeEditor::SpawnBranchNode() {
    m_Nodes.emplace_back(GetNextId(), "Branch");
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "", PinType::Flow);
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "Condition", PinType::Bool);
    m_Nodes.back().Outputs.emplace_back(GetNextId(), "True", PinType::Flow);
    m_Nodes.back().Outputs.emplace_back(GetNextId(), "False", PinType::Flow);

    BuildNode(&m_Nodes.back());

    return &m_Nodes.back();
}

Node *NodeEditor::SpawnDoNNode() {
    m_Nodes.emplace_back(GetNextId(), "Do N");
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "Enter", PinType::Flow);
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "N", PinType::Int);
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "Reset", PinType::Flow);
    m_Nodes.back().Outputs.emplace_back(GetNextId(), "Exit", PinType::Flow);
    m_Nodes.back().Outputs.emplace_back(GetNextId(), "Counter", PinType::Int);

    BuildNode(&m_Nodes.back());

    return &m_Nodes.back();
}

Node *NodeEditor::SpawnOutputActionNode() {
    m_Nodes.emplace_back(GetNextId(), "OutputAction");
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "Sample", PinType::Float);
    m_Nodes.back().Outputs.emplace_back(GetNextId(), "Condition", PinType::Bool);
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "Event", PinType::Delegate);

    BuildNode(&m_Nodes.back());

    return &m_Nodes.back();
}

Node *NodeEditor::SpawnPrintStringNode() {
    m_Nodes.emplace_back(GetNextId(), "Print String");
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "", PinType::Flow);
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "In String", PinType::String);
    m_Nodes.back().Outputs.emplace_back(GetNextId(), "", PinType::Flow);

    BuildNode(&m_Nodes.back());

    return &m_Nodes.back();
}

Node *NodeEditor::SpawnMessageNode() {
    m_Nodes.emplace_back(GetNextId(), "", ImColor(128, 195, 248));
    m_Nodes.back().Type = NodeType::Simple;
    m_Nodes.back().Outputs.emplace_back(GetNextId(), "Message", PinType::String);

    BuildNode(&m_Nodes.back());

    return &m_Nodes.back();
}

Node *NodeEditor::SpawnSetTimerNode() {
    m_Nodes.emplace_back(GetNextId(), "Set Timer", ImColor(128, 195, 248));
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "", PinType::Flow);
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "Object", PinType::Object);
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "Function Name", PinType::Function);
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "Time", PinType::Float);
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "Looping", PinType::Bool);
    m_Nodes.back().Outputs.emplace_back(GetNextId(), "", PinType::Flow);

    BuildNode(&m_Nodes.back());

    return &m_Nodes.back();
}

Node *NodeEditor::SpawnLessNode() {
    m_Nodes.emplace_back(GetNextId(), "<", ImColor(128, 195, 248));
    m_Nodes.back().Type = NodeType::Simple;
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "", PinType::Float);
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "", PinType::Float);
    m_Nodes.back().Outputs.emplace_back(GetNextId(), "", PinType::Float);

    BuildNode(&m_Nodes.back());

    return &m_Nodes.back();
}

Node *NodeEditor::SpawnWeirdNode() {
    m_Nodes.emplace_back(GetNextId(), "o.O", ImColor(128, 195, 248));
    m_Nodes.back().Type = NodeType::Simple;
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "", PinType::Float);
    m_Nodes.back().Outputs.emplace_back(GetNextId(), "", PinType::Float);
    m_Nodes.back().Outputs.emplace_back(GetNextId(), "", PinType::Float);

    BuildNode(&m_Nodes.back());

    return &m_Nodes.back();
}

Node *NodeEditor::SpawnTraceByChannelNode() {
    m_Nodes.emplace_back(GetNextId(), "Single Line Trace by Channel", ImColor(255, 128, 64));
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "", PinType::Flow);
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "Start", PinType::Flow);
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "End", PinType::Int);
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "Trace Channel", PinType::Float);
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "Trace Complex", PinType::Bool);
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "Actors to Ignore", PinType::Int);
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "Draw Debug Type", PinType::Bool);
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "Ignore Self", PinType::Bool);
    m_Nodes.back().Outputs.emplace_back(GetNextId(), "", PinType::Flow);
    m_Nodes.back().Outputs.emplace_back(GetNextId(), "Out Hit", PinType::Float);
    m_Nodes.back().Outputs.emplace_back(GetNextId(), "Return Value", PinType::Bool);

    BuildNode(&m_Nodes.back());

    return &m_Nodes.back();
}

Node *NodeEditor::SpawnTreeSequenceNode() {
    m_Nodes.emplace_back(GetNextId(), "Sequence");
    m_Nodes.back().Type = NodeType::Tree;
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "", PinType::Flow);
    m_Nodes.back().Outputs.emplace_back(GetNextId(), "", PinType::Flow);

    BuildNode(&m_Nodes.back());

    return &m_Nodes.back();
}

Node *NodeEditor::SpawnTreeTaskNode() {
    m_Nodes.emplace_back(GetNextId(), "Move To");
    m_Nodes.back().Type = NodeType::Tree;
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "", PinType::Flow);

    BuildNode(&m_Nodes.back());

    return &m_Nodes.back();
}

Node *NodeEditor::SpawnTreeTask2Node() {
    m_Nodes.emplace_back(GetNextId(), "Random Wait");
    m_Nodes.back().Type = NodeType::Tree;
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "", PinType::Flow);

    BuildNode(&m_Nodes.back());

    return &m_Nodes.back();
}

Node *NodeEditor::SpawnComment() {
    m_Nodes.emplace_back(GetNextId(), "Test Comment");
    m_Nodes.back().Type = NodeType::Comment;
    m_Nodes.back().Size = ImVec2(300, 200);

    return &m_Nodes.back();
}

Node *NodeEditor::SpawnHoudiniTransformNode() {
    m_Nodes.emplace_back(GetNextId(), "Transform");
    m_Nodes.back().Type = NodeType::Houdini;
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "", PinType::Flow);
    m_Nodes.back().Outputs.emplace_back(GetNextId(), "", PinType::Flow);

    BuildNode(&m_Nodes.back());

    return &m_Nodes.back();
}

Node *NodeEditor::SpawnHoudiniGroupNode() {
    m_Nodes.emplace_back(GetNextId(), "Group");
    m_Nodes.back().Type = NodeType::Houdini;
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "", PinType::Flow);
    m_Nodes.back().Inputs.emplace_back(GetNextId(), "", PinType::Flow);
    m_Nodes.back().Outputs.emplace_back(GetNextId(), "", PinType::Flow);

    BuildNode(&m_Nodes.back());

    return &m_Nodes.back();
}

void NodeEditor::init() {
    imgui_node_editor::Config config;
    config.SettingsFile = "Simple.json";
    imgui_node_editor_ctx = imgui_node_editor::CreateEditor(&config);

    imgui_node_editor::SetCurrentEditor(imgui_node_editor_ctx);
    Node *node;
    node = SpawnInputActionNode();
    imgui_node_editor::SetNodePosition(node->ID, ImVec2(-252, 220));
    node = SpawnBranchNode();
    imgui_node_editor::SetNodePosition(node->ID, ImVec2(-300, 351));
    node = SpawnDoNNode();
    imgui_node_editor::SetNodePosition(node->ID, ImVec2(-238, 504));
    node = SpawnOutputActionNode();
    imgui_node_editor::SetNodePosition(node->ID, ImVec2(71, 80));
    node = SpawnSetTimerNode();
    imgui_node_editor::SetNodePosition(node->ID, ImVec2(168, 316));

    node = SpawnTreeSequenceNode();
    imgui_node_editor::SetNodePosition(node->ID, ImVec2(1028, 329));
    node = SpawnTreeTaskNode();
    imgui_node_editor::SetNodePosition(node->ID, ImVec2(1204, 458));
    node = SpawnTreeTask2Node();
    imgui_node_editor::SetNodePosition(node->ID, ImVec2(868, 538));

    node = SpawnComment();
    imgui_node_editor::SetNodePosition(node->ID, ImVec2(112, 576));
    imgui_node_editor::SetGroupSize(node->ID, ImVec2(384, 154));
    node = SpawnComment();
    imgui_node_editor::SetNodePosition(node->ID, ImVec2(800, 224));
    imgui_node_editor::SetGroupSize(node->ID, ImVec2(640, 400));

    node = SpawnLessNode();
    imgui_node_editor::SetNodePosition(node->ID, ImVec2(366, 652));
    node = SpawnWeirdNode();
    imgui_node_editor::SetNodePosition(node->ID, ImVec2(144, 652));
    node = SpawnMessageNode();
    imgui_node_editor::SetNodePosition(node->ID, ImVec2(-348, 698));
    node = SpawnPrintStringNode();
    imgui_node_editor::SetNodePosition(node->ID, ImVec2(-69, 652));

    node = SpawnHoudiniTransformNode();
    imgui_node_editor::SetNodePosition(node->ID, ImVec2(500, -70));
    node = SpawnHoudiniGroupNode();
    imgui_node_editor::SetNodePosition(node->ID, ImVec2(500, 42));

    imgui_node_editor::NavigateToContent();
    BuildNodes();

    m_Links.push_back(Link(GetNextLinkId(), m_Nodes[5].Outputs[0].ID, m_Nodes[6].Inputs[0].ID));
    m_Links.push_back(Link(GetNextLinkId(), m_Nodes[5].Outputs[0].ID, m_Nodes[7].Inputs[0].ID));

    m_Links.push_back(Link(GetNextLinkId(), m_Nodes[14].Outputs[0].ID, m_Nodes[15].Inputs[0].ID));
    imgui_node_editor::SetCurrentEditor(nullptr);
}

void NodeEditor::deinit() {
}

void NodeEditor::update() {
    ImGui::Begin("Node Editor", nullptr, ImGuiWindowFlags_NoTitleBar);

    imgui_node_editor::SetCurrentEditor(imgui_node_editor_ctx);
    imgui_node_editor::Begin("My Editor", ImVec2(0.0, 0.0f));

    auto cursorTopLeft = ImGui::GetCursorScreenPos();

    static imgui_node_editor::NodeId contextNodeId = 0;
    static imgui_node_editor::LinkId contextLinkId = 0;
    static imgui_node_editor::PinId contextPinId = 0;
    static bool createNewNode = false;
    static Pin *newNodeLinkPin = nullptr;
    static Pin *newLinkPin = nullptr;

    auto builder = BlueprintNodeBuilder(m_HeaderBackground);

    for (auto &node : m_Nodes) {
        if (node.Type != NodeType::Blueprint && node.Type != NodeType::Simple)
            continue;

        bool hasOutputDelegates = false;
        for (auto &output : node.Outputs) {
            if (output.Type == PinType::Delegate) {
                hasOutputDelegates = true;
            }
        }

        const auto isSimple = node.Type == NodeType::Simple;
        builder.Begin(node.ID);
        {
            if (!isSimple) {
                builder.Header(node.Color);
                // ImGui::Spring(0);
                ImGui::TextUnformatted(node.Name.c_str());
                // ImGui::Spring(1);
                ImGui::Dummy(ImVec2(0, 28));
                if (hasOutputDelegates) {
                    // ImGui::BeginVertical("delegates", ImVec2(0, 28));
                    // ImGui::Spring(1, 0);
                    for (auto &output : node.Outputs) {
                        if (output.Type != PinType::Delegate)
                            continue;

                        auto alpha = ImGui::GetStyle().Alpha;
                        if (newLinkPin && !CanCreateLink(newLinkPin, &output) && &output != newLinkPin)
                            alpha = alpha * (48.0f / 255.0f);

                        imgui_node_editor::BeginPin(output.ID, imgui_node_editor::PinKind::Output);
                        imgui_node_editor::PinPivotAlignment(ImVec2(1.0f, 0.5f));
                        imgui_node_editor::PinPivotSize(ImVec2(0, 0));
                        // ImGui::BeginHorizontal(output.ID.AsPointer());
                        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
                        if (!output.Name.empty()) {
                            ImGui::TextUnformatted(output.Name.c_str());
                            // ImGui::Spring(0);
                        }
                        DrawPinIcon(output, IsPinLinked(output.ID), static_cast<int>(alpha * 255), static_cast<float>(m_PinIconSize));
                        // ImGui::Spring(0, ImGui::GetStyle().ItemSpacing.x / 2);
                        // ImGui::EndHorizontal();
                        ImGui::PopStyleVar();
                        imgui_node_editor::EndPin();

                        // DrawItemRect(ImColor(255, 0, 0));
                    }
                    // ImGui::Spring(1, 0);
                    // ImGui::EndVertical();
                    // ImGui::Spring(0, ImGui::GetStyle().ItemSpacing.x / 2);
                } else {
                    // ImGui::Spring(0);
                }
                builder.EndHeader();
            }

            for (auto &input : node.Inputs) {
                auto alpha = ImGui::GetStyle().Alpha;
                if (newLinkPin && !CanCreateLink(newLinkPin, &input) && &input != newLinkPin)
                    alpha = alpha * (48.0f / 255.0f);

                builder.Input(input.ID);
                ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
                DrawPinIcon(input, IsPinLinked(input.ID), static_cast<int>(alpha * 255), static_cast<float>(m_PinIconSize));
                // ImGui::Spring(0);
                if (!input.Name.empty()) {
                    ImGui::TextUnformatted(input.Name.c_str());
                    // ImGui::Spring(0);
                }
                if (input.Type == PinType::Bool) {
                    ImGui::Button("Hello");
                    // ImGui::Spring(0);
                }
                ImGui::PopStyleVar();
                builder.EndInput();
            }

            if (isSimple) {
                builder.Middle();
                // ImGui::Spring(1, 0);
                ImGui::TextUnformatted(node.Name.c_str());
                // ImGui::Spring(1, 0);
            }

            for (auto &output : node.Outputs) {
                if (!isSimple && output.Type == PinType::Delegate)
                    continue;

                auto alpha = ImGui::GetStyle().Alpha;
                if (newLinkPin && !CanCreateLink(newLinkPin, &output) && &output != newLinkPin)
                    alpha = alpha * (48.0f / 255.0f);

                ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
                builder.Output(output.ID);
                if (output.Type == PinType::String) {
                    static char buffer[128] = "Edit Me\nMultiline!";
                    static bool wasActive = false;

                    ImGui::PushItemWidth(100.0f);
                    ImGui::InputText("##edit", buffer, 127);
                    ImGui::PopItemWidth();
                    if (ImGui::IsItemActive() && !wasActive) {
                        imgui_node_editor::EnableShortcuts(false);
                        wasActive = true;
                    } else if (!ImGui::IsItemActive() && wasActive) {
                        imgui_node_editor::EnableShortcuts(true);
                        wasActive = false;
                    }
                    // ImGui::Spring(0);
                }
                if (!output.Name.empty()) {
                    // ImGui::Spring(0);
                    ImGui::TextUnformatted(output.Name.c_str());
                }
                // ImGui::Spring(0);
                DrawPinIcon(output, IsPinLinked(output.ID), static_cast<int>(alpha * 255), static_cast<float>(m_PinIconSize));
                ImGui::PopStyleVar();
                builder.EndOutput();
            }
        }
        builder.End();
    }

    for (auto &node : m_Nodes) {
        if (node.Type != NodeType::Tree)
            continue;

        const float rounding = 5.0f;
        const float padding = 12.0f;

        const auto pinBackground = imgui_node_editor::GetStyle().Colors[imgui_node_editor::StyleColor_NodeBg];

        imgui_node_editor::PushStyleColor(imgui_node_editor::StyleColor_NodeBg, ImColor(128, 128, 128, 200));
        imgui_node_editor::PushStyleColor(imgui_node_editor::StyleColor_NodeBorder, ImColor(32, 32, 32, 200));
        imgui_node_editor::PushStyleColor(imgui_node_editor::StyleColor_PinRect, ImColor(60, 180, 255, 150));
        imgui_node_editor::PushStyleColor(imgui_node_editor::StyleColor_PinRectBorder, ImColor(60, 180, 255, 150));

        imgui_node_editor::PushStyleVar(imgui_node_editor::StyleVar_NodePadding, ImVec4(0, 0, 0, 0));
        imgui_node_editor::PushStyleVar(imgui_node_editor::StyleVar_NodeRounding, rounding);
        imgui_node_editor::PushStyleVar(imgui_node_editor::StyleVar_SourceDirection, ImVec2(0.0f, 1.0f));
        imgui_node_editor::PushStyleVar(imgui_node_editor::StyleVar_TargetDirection, ImVec2(0.0f, -1.0f));
        imgui_node_editor::PushStyleVar(imgui_node_editor::StyleVar_LinkStrength, 0.0f);
        imgui_node_editor::PushStyleVar(imgui_node_editor::StyleVar_PinBorderWidth, 1.0f);
        imgui_node_editor::PushStyleVar(imgui_node_editor::StyleVar_PinRadius, 5.0f);
        imgui_node_editor::BeginNode(node.ID);

        // ImGui::BeginVertical(node.ID.AsPointer());
        // ImGui::BeginHorizontal("inputs");
        // ImGui::Spring(0, padding * 2);

        ImRect inputsRect;
        int inputAlpha = 200;
        if (!node.Inputs.empty()) {
            auto &pin = node.Inputs[0];
            ImGui::Dummy(ImVec2(0, padding));
            // ImGui::Spring(1, 0);
            // inputsRect = ImGui_GetItemRect();

            imgui_node_editor::PushStyleVar(imgui_node_editor::StyleVar_PinArrowSize, 10.0f);
            imgui_node_editor::PushStyleVar(imgui_node_editor::StyleVar_PinArrowWidth, 10.0f);
            imgui_node_editor::PushStyleVar(imgui_node_editor::StyleVar_PinCorners, ImDrawFlags_RoundCornersBottom);
            imgui_node_editor::BeginPin(pin.ID, imgui_node_editor::PinKind::Input);
            imgui_node_editor::PinPivotRect(inputsRect.GetTL(), inputsRect.GetBR());
            imgui_node_editor::PinRect(inputsRect.GetTL(), inputsRect.GetBR());
            imgui_node_editor::EndPin();
            imgui_node_editor::PopStyleVar(3);

            if (newLinkPin && !CanCreateLink(newLinkPin, &pin) && &pin != newLinkPin)
                inputAlpha = static_cast<int>(255 * ImGui::GetStyle().Alpha * (48.0f / 255.0f));
        } else {
            ImGui::Dummy(ImVec2(0, padding));
        }

        // ImGui::Spring(0, padding * 2);
        // ImGui::EndHorizontal();

        // ImGui::BeginHorizontal("content_frame");
        // ImGui::Spring(1, padding);

        // ImGui::BeginVertical("content", ImVec2(0.0f, 0.0f));
        // ImGui::Dummy(ImVec2(160, 0));
        // ImGui::Spring(1);
        // ImGui::TextUnformatted(node.Name.c_str());
        // ImGui::Spring(1);
        // ImGui::EndVertical();
        auto contentRect = ImGui_GetItemRect();

        // ImGui::Spring(1, padding);
        // ImGui::EndHorizontal();

        // ImGui::BeginHorizontal("outputs");
        // ImGui::Spring(0, padding * 2);

        ImRect outputsRect;
        int outputAlpha = 200;
        if (!node.Outputs.empty()) {
            auto &pin = node.Outputs[0];
            ImGui::Dummy(ImVec2(0, padding));
            // ImGui::Spring(1, 0);
            // outputsRect = ImGui_GetItemRect();

            imgui_node_editor::PushStyleVar(imgui_node_editor::StyleVar_PinCorners, ImDrawFlags_RoundCornersTop);
            imgui_node_editor::BeginPin(pin.ID, imgui_node_editor::PinKind::Output);
            imgui_node_editor::PinPivotRect(outputsRect.GetTL(), outputsRect.GetBR());
            imgui_node_editor::PinRect(outputsRect.GetTL(), outputsRect.GetBR());
            imgui_node_editor::EndPin();
            imgui_node_editor::PopStyleVar();

            if (newLinkPin && !CanCreateLink(newLinkPin, &pin) && &pin != newLinkPin)
                outputAlpha = static_cast<int>(255 * ImGui::GetStyle().Alpha * (48.0f / 255.0f));
        } else
            ImGui::Dummy(ImVec2(0, padding));

        // ImGui::Spring(0, padding * 2);
        // ImGui::EndHorizontal();

        // ImGui::EndVertical();

        imgui_node_editor::EndNode();
        imgui_node_editor::PopStyleVar(7);
        imgui_node_editor::PopStyleColor(4);

        auto drawList = imgui_node_editor::GetNodeBackgroundDrawList(node.ID);

        // const auto fringeScale = ImGui::GetStyle().AntiAliasFringeScale;
        // const auto unitSize    = 1.0f / fringeScale;

        // const auto ImDrawList_AddRect = [](ImDrawList* drawList, const ImVec2& a, const ImVec2& b, ImU32 col, float rounding, int rounding_corners, float thickness)
        //{
        //     if ((col >> 24) == 0)
        //         return;
        //     drawList->PathRect(a, b, rounding, rounding_corners);
        //     drawList->PathStroke(col, true, thickness);
        // };

        const auto topRoundCornersFlags = ImDrawFlags_RoundCornersTop;
        const auto bottomRoundCornersFlags = ImDrawFlags_RoundCornersBottom;

        drawList->AddRectFilled(inputsRect.GetTL() + ImVec2(0, 1), inputsRect.GetBR(),
                                IM_COL32(static_cast<int>(255 * pinBackground.x), static_cast<int>(255 * pinBackground.y), static_cast<int>(255 * pinBackground.z), inputAlpha), 4.0f, bottomRoundCornersFlags);
        // ImGui::PushStyleVar(ImGuiStyleVar_AntiAliasFringeScale, 1.0f);
        drawList->AddRect(inputsRect.GetTL() + ImVec2(0, 1), inputsRect.GetBR(),
                          IM_COL32(static_cast<int>(255 * pinBackground.x), static_cast<int>(255 * pinBackground.y), static_cast<int>(255 * pinBackground.z), inputAlpha), 4.0f, bottomRoundCornersFlags);
        // ImGui::PopStyleVar();
        drawList->AddRectFilled(outputsRect.GetTL(), outputsRect.GetBR() - ImVec2(0, 1),
                                IM_COL32(static_cast<int>(255 * pinBackground.x), static_cast<int>(255 * pinBackground.y), static_cast<int>(255 * pinBackground.z), outputAlpha), 4.0f, topRoundCornersFlags);
        // ImGui::PushStyleVar(ImGuiStyleVar_AntiAliasFringeScale, 1.0f);
        drawList->AddRect(outputsRect.GetTL(), outputsRect.GetBR() - ImVec2(0, 1),
                          IM_COL32(static_cast<int>(255 * pinBackground.x), static_cast<int>(255 * pinBackground.y), static_cast<int>(255 * pinBackground.z), outputAlpha), 4.0f, topRoundCornersFlags);
        // ImGui::PopStyleVar();
        drawList->AddRectFilled(contentRect.GetTL(), contentRect.GetBR(), IM_COL32(24, 64, 128, 200), 0.0f);
        // ImGui::PushStyleVar(ImGuiStyleVar_AntiAliasFringeScale, 1.0f);
        drawList->AddRect(
            contentRect.GetTL(),
            contentRect.GetBR(),
            IM_COL32(48, 128, 255, 100), 0.0f);
        // ImGui::PopStyleVar();
    }

    for (auto &node : m_Nodes) {
        if (node.Type != NodeType::Houdini)
            continue;

        const float rounding = 10.0f;
        const float padding = 12.0f;

        imgui_node_editor::PushStyleColor(imgui_node_editor::StyleColor_NodeBg, ImColor(229, 229, 229, 200));
        imgui_node_editor::PushStyleColor(imgui_node_editor::StyleColor_NodeBorder, ImColor(125, 125, 125, 200));
        imgui_node_editor::PushStyleColor(imgui_node_editor::StyleColor_PinRect, ImColor(229, 229, 229, 60));
        imgui_node_editor::PushStyleColor(imgui_node_editor::StyleColor_PinRectBorder, ImColor(125, 125, 125, 60));

        const auto pinBackground = imgui_node_editor::GetStyle().Colors[imgui_node_editor::StyleColor_NodeBg];

        imgui_node_editor::PushStyleVar(imgui_node_editor::StyleVar_NodePadding, ImVec4(0, 0, 0, 0));
        imgui_node_editor::PushStyleVar(imgui_node_editor::StyleVar_NodeRounding, rounding);
        imgui_node_editor::PushStyleVar(imgui_node_editor::StyleVar_SourceDirection, ImVec2(0.0f, 1.0f));
        imgui_node_editor::PushStyleVar(imgui_node_editor::StyleVar_TargetDirection, ImVec2(0.0f, -1.0f));
        imgui_node_editor::PushStyleVar(imgui_node_editor::StyleVar_LinkStrength, 0.0f);
        imgui_node_editor::PushStyleVar(imgui_node_editor::StyleVar_PinBorderWidth, 1.0f);
        imgui_node_editor::PushStyleVar(imgui_node_editor::StyleVar_PinRadius, 6.0f);
        imgui_node_editor::BeginNode(node.ID);

        // ImGui::BeginVertical(node.ID.AsPointer());
        if (!node.Inputs.empty()) {
            // ImGui::BeginHorizontal("inputs");
            // ImGui::Spring(1, 0);

            ImRect inputsRect;
            int inputAlpha = 200;
            for (auto &pin : node.Inputs) {
                ImGui::Dummy(ImVec2(padding, padding));
                inputsRect = ImGui_GetItemRect();
                // ImGui::Spring(1, 0);
                inputsRect.Min.y -= padding;
                inputsRect.Max.y -= padding;

                const auto allRoundCornersFlags = ImDrawFlags_RoundCornersAll;
                // imgui_node_editor::PushStyleVar(imgui_node_editor::StyleVar_PinArrowSize, 10.0f);
                // imgui_node_editor::PushStyleVar(imgui_node_editor::StyleVar_PinArrowWidth, 10.0f);
                imgui_node_editor::PushStyleVar(imgui_node_editor::StyleVar_PinCorners, allRoundCornersFlags);

                imgui_node_editor::BeginPin(pin.ID, imgui_node_editor::PinKind::Input);
                imgui_node_editor::PinPivotRect(inputsRect.GetCenter(), inputsRect.GetCenter());
                imgui_node_editor::PinRect(inputsRect.GetTL(), inputsRect.GetBR());
                imgui_node_editor::EndPin();
                // imgui_node_editor::PopStyleVar(3);
                imgui_node_editor::PopStyleVar(1);

                auto drawList = ImGui::GetWindowDrawList();
                drawList->AddRectFilled(inputsRect.GetTL(), inputsRect.GetBR(),
                                        IM_COL32(static_cast<int>(255 * pinBackground.x), static_cast<int>(255 * pinBackground.y), static_cast<int>(255 * pinBackground.z), inputAlpha), 4.0f, allRoundCornersFlags);
                drawList->AddRect(inputsRect.GetTL(), inputsRect.GetBR(),
                                  IM_COL32(static_cast<int>(255 * pinBackground.x), static_cast<int>(255 * pinBackground.y), static_cast<int>(255 * pinBackground.z), inputAlpha), 4.0f, allRoundCornersFlags);

                if (newLinkPin && !CanCreateLink(newLinkPin, &pin) && &pin != newLinkPin)
                    inputAlpha = static_cast<int>(255 * ImGui::GetStyle().Alpha * (48.0f / 255.0f));
            }

            // ImGui::Spring(1, 0);
            //  ImGui::EndHorizontal();
        }

        // ImGui::BeginHorizontal("content_frame");
        // ImGui::Spring(1, padding);

        // ImGui::BeginVertical("content", ImVec2(0.0f, 0.0f));
        ImGui::Dummy(ImVec2(160, 0));
        // ImGui::Spring(1);
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
        ImGui::TextUnformatted(node.Name.c_str());
        ImGui::PopStyleColor();
        // ImGui::Spring(1);
        // ImGui::EndVertical();
        // auto contentRect = ImGui_GetItemRect();

        // ImGui::Spring(1, padding);
        // ImGui::EndHorizontal();

        if (!node.Outputs.empty()) {
            // ImGui::BeginHorizontal("outputs");
            // ImGui::Spring(1, 0);

            ImRect outputsRect;
            int outputAlpha = 200;
            for (auto &pin : node.Outputs) {
                ImGui::Dummy(ImVec2(padding, padding));
                outputsRect = ImGui_GetItemRect();
                // ImGui::Spring(1, 0);
                outputsRect.Min.y += padding;
                outputsRect.Max.y += padding;

#if IMGUI_VERSION_NUM > 18101
                const auto allRoundCornersFlags = ImDrawFlags_RoundCornersAll;
                const auto topRoundCornersFlags = ImDrawFlags_RoundCornersTop;
#else
                const auto allRoundCornersFlags = 15;
                const auto topRoundCornersFlags = 3;
#endif

                imgui_node_editor::PushStyleVar(imgui_node_editor::StyleVar_PinCorners, topRoundCornersFlags);
                imgui_node_editor::BeginPin(pin.ID, imgui_node_editor::PinKind::Output);
                imgui_node_editor::PinPivotRect(outputsRect.GetCenter(), outputsRect.GetCenter());
                imgui_node_editor::PinRect(outputsRect.GetTL(), outputsRect.GetBR());
                imgui_node_editor::EndPin();
                imgui_node_editor::PopStyleVar();

                auto drawList = ImGui::GetWindowDrawList();
                drawList->AddRectFilled(outputsRect.GetTL(), outputsRect.GetBR(),
                                        IM_COL32(static_cast<int>(255 * pinBackground.x), static_cast<int>(255 * pinBackground.y), static_cast<int>(255 * pinBackground.z), outputAlpha), 4.0f, allRoundCornersFlags);
                drawList->AddRect(outputsRect.GetTL(), outputsRect.GetBR(),
                                  IM_COL32(static_cast<int>(255 * pinBackground.x), static_cast<int>(255 * pinBackground.y), static_cast<int>(255 * pinBackground.z), outputAlpha), 4.0f, allRoundCornersFlags);

                if (newLinkPin && !CanCreateLink(newLinkPin, &pin) && &pin != newLinkPin)
                    outputAlpha = static_cast<int>(255 * ImGui::GetStyle().Alpha * (48.0f / 255.0f));
            }

            // ImGui::EndHorizontal();
        }

        // ImGui::EndVertical();

        imgui_node_editor::EndNode();
        imgui_node_editor::PopStyleVar(7);
        imgui_node_editor::PopStyleColor(4);
    }

    for (auto &node : m_Nodes) {
        if (node.Type != NodeType::Comment)
            continue;

        const float commentAlpha = 0.75f;

        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, commentAlpha);
        imgui_node_editor::PushStyleColor(imgui_node_editor::StyleColor_NodeBg, ImColor(255, 255, 255, 64));
        imgui_node_editor::PushStyleColor(imgui_node_editor::StyleColor_NodeBorder, ImColor(255, 255, 255, 64));
        imgui_node_editor::BeginNode(node.ID);
        ImGui::PushID(node.ID.AsPointer());
        // ImGui::BeginVertical("content");
        // ImGui::BeginHorizontal("horizontal");
        // ImGui::Spring(1);
        ImGui::TextUnformatted(node.Name.c_str());
        // ImGui::Spring(1);
        // ImGui::EndHorizontal();
        imgui_node_editor::Group(node.Size);
        // ImGui::EndVertical();
        ImGui::PopID();
        imgui_node_editor::EndNode();
        imgui_node_editor::PopStyleColor(2);
        ImGui::PopStyleVar();

        if (imgui_node_editor::BeginGroupHint(node.ID)) {
            // auto alpha   = static_cast<int>(commentAlpha * ImGui::GetStyle().Alpha * 255);
            auto bgAlpha = static_cast<int>(ImGui::GetStyle().Alpha * 255);

            // ImGui::PushStyleVar(ImGuiStyleVar_Alpha, commentAlpha * ImGui::GetStyle().Alpha);

            auto min = imgui_node_editor::GetGroupMin();
            // auto max = imgui_node_editor::GetGroupMax();

            ImGui::SetCursorScreenPos(min - ImVec2(-8, ImGui::GetTextLineHeightWithSpacing() + 4));
            ImGui::BeginGroup();
            ImGui::TextUnformatted(node.Name.c_str());
            ImGui::EndGroup();

            auto drawList = imgui_node_editor::GetHintBackgroundDrawList();

            auto hintBounds = ImGui_GetItemRect();
            auto hintFrameBounds = ImRect_Expanded(hintBounds, 8, 4);

            drawList->AddRectFilled(
                hintFrameBounds.GetTL(),
                hintFrameBounds.GetBR(),
                IM_COL32(255, 255, 255, 64 * bgAlpha / 255), 4.0f);

            drawList->AddRect(
                hintFrameBounds.GetTL(),
                hintFrameBounds.GetBR(),
                IM_COL32(255, 255, 255, 128 * bgAlpha / 255), 4.0f);

            // ImGui::PopStyleVar();
        }
        imgui_node_editor::EndGroupHint();
    }

    for (auto &link : m_Links) {
        imgui_node_editor::Link(link.ID, link.StartPinID, link.EndPinID, link.Color, 2.0f);
    }

    if (!createNewNode) {
        if (imgui_node_editor::BeginCreate(ImColor(255, 255, 255), 2.0f)) {
            auto showLabel = [](const char *label, ImColor color) {
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() - ImGui::GetTextLineHeight());
                auto size = ImGui::CalcTextSize(label);

                auto padding = ImGui::GetStyle().FramePadding;
                auto spacing = ImGui::GetStyle().ItemSpacing;

                ImGui::SetCursorPos(ImGui::GetCursorPos() + ImVec2(spacing.x, -spacing.y));

                auto rectMin = ImGui::GetCursorScreenPos() - padding;
                auto rectMax = ImGui::GetCursorScreenPos() + size + padding;

                auto drawList = ImGui::GetWindowDrawList();
                drawList->AddRectFilled(rectMin, rectMax, color, size.y * 0.15f);
                ImGui::TextUnformatted(label);
            };

            imgui_node_editor::PinId startPinId = 0, endPinId = 0;
            if (imgui_node_editor::QueryNewLink(&startPinId, &endPinId)) {
                auto startPin = FindPin(startPinId);
                auto endPin = FindPin(endPinId);

                newLinkPin = startPin ? startPin : endPin;

                if (startPin->Kind == PinKind::Input) {
                    std::swap(startPin, endPin);
                    std::swap(startPinId, endPinId);
                }

                if (startPin && endPin) {
                    if (endPin == startPin) {
                        imgui_node_editor::RejectNewItem(ImColor(255, 0, 0), 2.0f);
                    } else if (endPin->Kind == startPin->Kind) {
                        showLabel("x Incompatible Pin Kind", ImColor(45, 32, 32, 180));
                        imgui_node_editor::RejectNewItem(ImColor(255, 0, 0), 2.0f);
                    }
                    // else if (endPin->Node == startPin->Node)
                    //{
                    //     showLabel("x Cannot connect to self", ImColor(45, 32, 32, 180));
                    //     imgui_node_editor::RejectNewItem(ImColor(255, 0, 0), 1.0f);
                    // }
                    else if (endPin->Type != startPin->Type) {
                        showLabel("x Incompatible Pin Type", ImColor(45, 32, 32, 180));
                        imgui_node_editor::RejectNewItem(ImColor(255, 128, 128), 1.0f);
                    } else {
                        showLabel("+ Create Link", ImColor(32, 45, 32, 180));
                        if (imgui_node_editor::AcceptNewItem(ImColor(128, 255, 128), 4.0f)) {
                            m_Links.emplace_back(Link(static_cast<imgui_node_editor::LinkId>(static_cast<uint32_t>(GetNextId())), startPinId, endPinId));
                            m_Links.back().Color = GetIconColor(startPin->Type);
                        }
                    }
                }
            }

            imgui_node_editor::PinId pinId = 0;
            if (imgui_node_editor::QueryNewNode(&pinId)) {
                newLinkPin = FindPin(pinId);
                if (newLinkPin)
                    showLabel("+ Create Node", ImColor(32, 45, 32, 180));

                if (imgui_node_editor::AcceptNewItem()) {
                    createNewNode = true;
                    newNodeLinkPin = FindPin(pinId);
                    newLinkPin = nullptr;
                    imgui_node_editor::Suspend();
                    ImGui::OpenPopup("Create New Node");
                    imgui_node_editor::Resume();
                }
            }
        } else
            newLinkPin = nullptr;

        imgui_node_editor::EndCreate();

        if (imgui_node_editor::BeginDelete()) {
            imgui_node_editor::LinkId linkId = 0;
            while (imgui_node_editor::QueryDeletedLink(&linkId)) {
                if (imgui_node_editor::AcceptDeletedItem()) {
                    auto id = std::find_if(m_Links.begin(), m_Links.end(), [linkId](auto &link) { return link.ID == linkId; });
                    if (id != m_Links.end())
                        m_Links.erase(id);
                }
            }

            imgui_node_editor::NodeId nodeId = 0;
            while (imgui_node_editor::QueryDeletedNode(&nodeId)) {
                if (imgui_node_editor::AcceptDeletedItem()) {
                    auto id = std::find_if(m_Nodes.begin(), m_Nodes.end(), [nodeId](auto &node) { return node.ID == nodeId; });
                    if (id != m_Nodes.end())
                        m_Nodes.erase(id);
                }
            }
        }
        imgui_node_editor::EndDelete();
    }

    ImGui::SetCursorScreenPos(cursorTopLeft);

    {
        auto openPopupPosition = ImGui::GetMousePos();
        imgui_node_editor::Suspend();
        if (imgui_node_editor::ShowNodeContextMenu(&contextNodeId))
            ImGui::OpenPopup("Node Context Menu");
        else if (imgui_node_editor::ShowPinContextMenu(&contextPinId))
            ImGui::OpenPopup("Pin Context Menu");
        else if (imgui_node_editor::ShowLinkContextMenu(&contextLinkId))
            ImGui::OpenPopup("Link Context Menu");
        else if (imgui_node_editor::ShowBackgroundContextMenu()) {
            ImGui::OpenPopup("Create New Node");
            newNodeLinkPin = nullptr;
        }
        imgui_node_editor::Resume();

        imgui_node_editor::Suspend();
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));
        if (ImGui::BeginPopup("Node Context Menu")) {
            auto node = FindNode(contextNodeId);

            ImGui::TextUnformatted("Node Context Menu");
            ImGui::Separator();
            if (node) {
                ImGui::Text("ID: %p", node->ID.AsPointer());
                ImGui::Text("Type: %s", node->Type == NodeType::Blueprint ? "Blueprint" : (node->Type == NodeType::Tree ? "Tree" : "Comment"));
                ImGui::Text("Inputs: %d", static_cast<int>(node->Inputs.size()));
                ImGui::Text("Outputs: %d", static_cast<int>(node->Outputs.size()));
            } else
                ImGui::Text("Unknown node: %p", contextNodeId.AsPointer());
            ImGui::Separator();
            if (ImGui::MenuItem("Delete"))
                imgui_node_editor::DeleteNode(contextNodeId);
            ImGui::EndPopup();
        }

        if (ImGui::BeginPopup("Pin Context Menu")) {
            auto pin = FindPin(contextPinId);

            ImGui::TextUnformatted("Pin Context Menu");
            ImGui::Separator();
            if (pin) {
                ImGui::Text("ID: %p", pin->ID.AsPointer());
                if (pin->Node)
                    ImGui::Text("Node: %p", pin->Node->ID.AsPointer());
                else
                    ImGui::Text("Node: %s", "<none>");
            } else
                ImGui::Text("Unknown pin: %p", contextPinId.AsPointer());

            ImGui::EndPopup();
        }

        if (ImGui::BeginPopup("Link Context Menu")) {
            auto link = FindLink(contextLinkId);

            ImGui::TextUnformatted("Link Context Menu");
            ImGui::Separator();
            if (link) {
                ImGui::Text("ID: %p", link->ID.AsPointer());
                ImGui::Text("From: %p", link->StartPinID.AsPointer());
                ImGui::Text("To: %p", link->EndPinID.AsPointer());
            } else
                ImGui::Text("Unknown link: %p", contextLinkId.AsPointer());
            ImGui::Separator();
            if (ImGui::MenuItem("Delete"))
                imgui_node_editor::DeleteLink(contextLinkId);
            ImGui::EndPopup();
        }

        if (ImGui::BeginPopup("Create New Node")) {
            auto newNodePostion = openPopupPosition;
            // ImGui::SetCursorScreenPos(ImGui::GetMousePosOnOpeningCurrentPopup());

            // auto drawList = ImGui::GetWindowDrawList();
            // drawList->AddCircleFilled(ImGui::GetMousePosOnOpeningCurrentPopup(), 10.0f, 0xFFFF00FF);

            Node *node = nullptr;
            if (ImGui::MenuItem("Input Action"))
                node = SpawnInputActionNode();
            if (ImGui::MenuItem("Output Action"))
                node = SpawnOutputActionNode();
            if (ImGui::MenuItem("Branch"))
                node = SpawnBranchNode();
            if (ImGui::MenuItem("Do N"))
                node = SpawnDoNNode();
            if (ImGui::MenuItem("Set Timer"))
                node = SpawnSetTimerNode();
            if (ImGui::MenuItem("Less"))
                node = SpawnLessNode();
            if (ImGui::MenuItem("Weird"))
                node = SpawnWeirdNode();
            if (ImGui::MenuItem("Trace by Channel"))
                node = SpawnTraceByChannelNode();
            if (ImGui::MenuItem("Print String"))
                node = SpawnPrintStringNode();
            ImGui::Separator();
            if (ImGui::MenuItem("Comment"))
                node = SpawnComment();
            ImGui::Separator();
            if (ImGui::MenuItem("Sequence"))
                node = SpawnTreeSequenceNode();
            if (ImGui::MenuItem("Move To"))
                node = SpawnTreeTaskNode();
            if (ImGui::MenuItem("Random Wait"))
                node = SpawnTreeTask2Node();
            ImGui::Separator();
            if (ImGui::MenuItem("Message"))
                node = SpawnMessageNode();
            ImGui::Separator();
            if (ImGui::MenuItem("Transform"))
                node = SpawnHoudiniTransformNode();
            if (ImGui::MenuItem("Group"))
                node = SpawnHoudiniGroupNode();

            if (node) {
                BuildNodes();

                createNewNode = false;

                imgui_node_editor::SetNodePosition(node->ID, newNodePostion);

                if (auto startPin = newNodeLinkPin) {
                    auto &pins = startPin->Kind == PinKind::Input ? node->Outputs : node->Inputs;

                    for (auto &pin : pins) {
                        if (CanCreateLink(startPin, &pin)) {
                            auto endPin = &pin;
                            if (startPin->Kind == PinKind::Input)
                                std::swap(startPin, endPin);

                            m_Links.emplace_back(Link(static_cast<imgui_node_editor::LinkId>(static_cast<uint32_t>(GetNextId())), startPin->ID, endPin->ID));
                            m_Links.back().Color = GetIconColor(startPin->Type);

                            break;
                        }
                    }
                }
            }

            ImGui::EndPopup();
        } else
            createNewNode = false;
        ImGui::PopStyleVar();
        imgui_node_editor::Resume();
    }

    imgui_node_editor::End();
    imgui_node_editor::SetCurrentEditor(nullptr);

    ImGui::End();
}
