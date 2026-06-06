import _uigraph as u


def _mini_graph():
    return {
        "last_node_id": 2, "last_link_id": 0, "nodes": [
            {"id": 1, "type": "LoadImage", "pos": [0, 0],
             "inputs": [], "outputs": [
                 {"name": "IMAGE", "type": "IMAGE", "links": []},
                 {"name": "MASK", "type": "MASK", "links": []}]},
            {"id": 2, "type": "InvertMask", "pos": [0, 0],
             "inputs": [{"name": "mask", "type": "MASK", "link": None}],
             "outputs": [{"name": "MASK", "type": "MASK", "links": []}]},
        ], "links": [],
    }


def test_out_slot_and_in_slot_by_name():
    g = _mini_graph()
    assert u.out_slot(u.find_by_type(g, "LoadImage"), "MASK") == 1
    assert u.in_slot(u.find_by_type(g, "InvertMask"), "mask") == 0


def test_add_link_wires_both_ends_and_validates():
    g = _mini_graph()
    u.add_link(g, 1, "MASK", 2, "mask", "MASK")
    link = g["links"][0]
    assert link[1] == 1 and link[3] == 2  # src id, dst id
    assert g["nodes"][1]["inputs"][0]["link"] == link[0]
    assert link[0] in u.find_by_type(g, "LoadImage")["outputs"][1]["links"]
    u.assert_graph_valid(g)  # no dangling endpoints


def test_assert_graph_valid_catches_dangling():
    g = _mini_graph()
    g["links"].append([99, 1, 5, 2, 0, "MASK"])  # src slot 5 doesn't exist
    try:
        u.assert_graph_valid(g)
    except AssertionError:
        return
    raise AssertionError("expected dangling link to be caught")


def test_add_node_allocates_id_and_increments():
    g = _mini_graph()
    nid = u.add_node(g, ntype="EmptyImage", title="white", pos=[10, 10],
                     outputs=[{"name": "IMAGE", "type": "IMAGE", "links": []}],
                     widgets=[512, 512, 1, 1.0])
    assert nid == 3 and g["last_node_id"] == 3
    assert u.find_by_type(g, "EmptyImage")["widgets_values"] == [512, 512, 1, 1.0]
