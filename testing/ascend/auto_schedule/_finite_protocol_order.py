"""Finite completion graph for the reviewed fixed synchronization fixtures."""

from collections import defaultdict


def prove(events, require_drain=True):
    edges = defaultdict(set)
    reasons = {}
    pipes = defaultdict(list)
    channels = defaultdict(lambda: {"set": [], "wait": []})

    def edge(source, destination, reason):
        edges[source].add(destination)
        reasons[source, destination] = reason

    for index, event in enumerate(events):
        edge(2 * index, 2 * index + 1, "completion follows issue")
        previous = pipes[event["core"], event["pipe"]]
        if previous:
            prior = previous[-1]
            edge(2 * prior, 2 * index, "physical pipeline issue order")
            if event["pipe"] in ("S", "V") or events[prior]["kind"] in ("set", "wait", "barrier"):
                edge(2 * prior + 1, 2 * index, "implicit completion or synchronization gate")
        if event["kind"] in ("set", "barrier"):
            for prior in previous:
                edge(2 * prior + 1, 2 * index, "producer release/barrier completes prior pipeline work")
        previous.append(index)
        if event["kind"] in ("set", "wait"):
            channels[tuple(event["channel"])][event["kind"]].append(index)

    errors = []
    for channel, uses in channels.items():
        if len(uses["set"]) != len(uses["wait"]):
            errors.append({"error": "unbalanced token", "channel": channel, **uses})
        for release, acquire in zip(uses["set"], uses["wait"]):
            edge(2 * release + 1, 2 * acquire + 1, "matching hardware token completion")

    reachable = {}
    active = set()

    def visit(vertex):
        assert vertex not in active, "token dependency cycle"
        if vertex not in reachable:
            active.add(vertex)
            result = set()
            for target in edges[vertex]:
                result.add(target)
                result.update(visit(target))
            active.remove(vertex)
            reachable[vertex] = result
        return reachable[vertex]

    cycle = None
    try:
        for vertex in range(2 * len(events)):
            visit(vertex)
    except AssertionError as error:
        cycle = str(error)

    def before(source, target):
        return cycle is None and target in reachable[source]

    for channel, uses in channels.items():
        for acquire, release in zip(uses["wait"], uses["set"][1:]):
            if not before(2 * acquire + 1, 2 * release):
                errors.append(
                    {"error": "binary token released again before acquire", "channel": channel, "acquire": acquire, "release": release}
                )

    def conflicts(left, right):
        if left["storage"] != right["storage"]:
            return left["scope"] == right["scope"] == "global"
        return bool({tuple(index) for index in left["indices"]} & {tuple(index) for index in right["indices"]})

    hazards, unordered = [], []
    memory = [(index, event) for index, event in enumerate(events) if event["kind"] == "memory"]
    for position, (left_id, left) in enumerate(memory):
        for right_id, right in memory[position + 1 :]:
            kinds = []
            for name, lhs, rhs in (
                ("RAW", left["writes"], right["reads"]),
                ("WAR", left["reads"], right["writes"]),
                ("WAW", left["writes"], right["writes"]),
            ):
                if any(conflicts(a, b) for a in lhs for b in rhs):
                    kinds.append(name)
            if not kinds:
                continue
            hazard = {"source": left_id, "destination": right_id, "kinds": kinds}
            hazards.append(hazard)
            if not before(2 * left_id + 1, 2 * right_id):
                unordered.append(hazard)

    tails = [index for index, event in enumerate(events) if event["kind"] in ("wait", "barrier") and not event["loops"]]
    copies = [index for index, event in memory if event["operation"] == "copy"]
    undrained = [index for index in copies if not any(before(2 * index + 1, 2 * tail + 1) for tail in tails)] if require_drain else []
    return {
        "safe": not (errors or cycle or unordered or undrained),
        "events": events,
        "edges": [[source, target, reasons[source, target]] for source in sorted(edges) for target in sorted(edges[source])],
        "channels": [{"key": list(key), **uses} for key, uses in channels.items()],
        "token_errors": errors,
        "cycle": cycle,
        "hazards": hazards,
        "unordered": unordered,
        "undrained": undrained,
        "drains": tails,
        "reachable": {str(vertex): sorted(targets) for vertex, targets in reachable.items()} if cycle is None else {},
    }
