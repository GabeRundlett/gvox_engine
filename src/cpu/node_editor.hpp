#pragma once

#include <array>
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <set>
#include <map>

#include <imgui_node_editor.h>

namespace imgui_node_editor = ax::NodeEditor;

enum class PinType {
    Flow,
    Bool,
    Int,
    Float,
    String,
    Object,
    Function,
    Delegate,
};

enum class PinKind {
    Output,
    Input,
};

enum class NodeType {
    Blueprint,
    Simple,
    Tree,
    Comment,
    Houdini,
};

struct Node;

struct Pin {
    imgui_node_editor::PinId ID;
    Node *Node;
    std::string Name;
    PinType Type;
    PinKind Kind;

    Pin(int id, const char *name, PinType type)
        : ID(static_cast<imgui_node_editor::PinId>(static_cast<uint32_t>(id))),
          Node(nullptr),
          Name(name),
          Type(type),
          Kind(PinKind::Input) {
    }
};

struct Node {
    imgui_node_editor::NodeId ID;
    std::string Name;
    std::vector<Pin> Inputs;
    std::vector<Pin> Outputs;
    ImColor Color;
    NodeType Type;
    ImVec2 Size;

    std::string State;
    std::string SavedState;

    Node(int id, const char *name, ImColor color = ImColor(255, 255, 255))
        : ID(static_cast<imgui_node_editor::NodeId>(static_cast<uint32_t>(id))),
          Name(name),
          Color(color),
          Type(NodeType::Blueprint),
          Size(0, 0) {
    }
};

struct Link {
    imgui_node_editor::LinkId ID;

    imgui_node_editor::PinId StartPinID;
    imgui_node_editor::PinId EndPinID;

    ImColor Color;

    Link(imgui_node_editor::LinkId id, imgui_node_editor::PinId startPinId, imgui_node_editor::PinId endPinId)
        : ID(id),
          StartPinID(startPinId),
          EndPinID(endPinId),
          Color(255, 255, 255) {
    }
};

struct NodeIdLess {
    bool operator()(const imgui_node_editor::NodeId &lhs, const imgui_node_editor::NodeId &rhs) const {
        return lhs.AsPointer() < rhs.AsPointer();
    }
};

struct NodeEditor {
    imgui_node_editor::EditorContext *imgui_node_editor_ctx = nullptr;

    int m_NextId = 1;
    const int m_PinIconSize = 24;
    std::vector<Node> m_Nodes;
    std::vector<Link> m_Links;
    ImTextureID m_HeaderBackground = nullptr;
    ImTextureID m_SaveIcon = nullptr;
    ImTextureID m_RestoreIcon = nullptr;
    const float m_TouchTime = 1.0f;
    std::map<imgui_node_editor::NodeId, float, NodeIdLess> m_NodeTouchTime;
    bool m_ShowOrdinals = false;

    void init();
    void deinit();
    void update();

    int GetNextId();
    imgui_node_editor::LinkId GetNextLinkId();

    Node *FindNode(imgui_node_editor::NodeId id);
    Link *FindLink(imgui_node_editor::LinkId id);
    Pin *FindPin(imgui_node_editor::PinId id);
    bool IsPinLinked(imgui_node_editor::PinId id);

    static void BuildNode(Node *node);
    void BuildNodes();

    Node *SpawnInputActionNode();
    Node *SpawnBranchNode();
    Node *SpawnDoNNode();
    Node *SpawnOutputActionNode();
    Node *SpawnPrintStringNode();
    Node *SpawnMessageNode();
    Node *SpawnSetTimerNode();
    Node *SpawnLessNode();
    Node *SpawnWeirdNode();
    Node *SpawnTraceByChannelNode();
    Node *SpawnTreeSequenceNode();
    Node *SpawnTreeTaskNode();
    Node *SpawnTreeTask2Node();
    Node *SpawnComment();
    Node *SpawnHoudiniTransformNode();
    Node *SpawnHoudiniGroupNode();
};
