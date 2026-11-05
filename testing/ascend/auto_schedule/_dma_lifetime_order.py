"""Finite event proof using physical pipe execution and real flag tokens only."""

from collections import defaultdict


def prove(events):
    """Check every a/b generation against every generation of the other storage.

    DMA issue order is deliberately weaker than DMA completion. A producer
    flag completes only after prior work on that pipe; a consumer wait gates
    later work. Binary event reuse additionally needs an acquire before the
    next release, and the last DMA needs a real outer drain.
    """
    adjacency = defaultdict(set)
    edge_records = []

    def edge(source, target, reason):
        if target not in adjacency[source]:
            adjacency[source].add(target)
            edge_records.append([source, target, reason])

    channels = defaultdict(lambda: {"set": [], "wait": []})
    pipes = defaultdict(list)
    for index, event in enumerate(events):
        event["id"] = index
        assert event["core"] == 5, ("The proof requires the actual common AIV mask", event)
        assert event["stage"] == 0, ("This proof uses zero-stage loops", event)
        edge(2 * index, 2 * index + 1, "operation completes after issue")
        previous = pipes[(event["core"], event["pipe"])]
        if previous:
            prior = previous[-1]
            edge(2 * prior, 2 * index, "same-pipe issue order")
            if event["pipe"] == "S" or events[prior]["kind"] in ("set", "wait"):
                edge(2 * prior + 1, 2 * index, "scalar completion or flag gate")
        if event["kind"] == "set":
            for prior in previous:
                edge(2 * prior + 1, 2 * index, "producer release completes previous pipe work")
        previous.append(index)
        if event["kind"] in ("set", "wait"):
            channels[(event["core"], event["channel"], event["token"])][event["kind"]].append(index)

    token_errors = []
    for channel, uses in channels.items():
        if len(uses["set"]) != len(uses["wait"]):
            token_errors.append({"channel": channel, "error": "unbalanced tokens", **uses})
        for release, acquire in zip(uses["set"], uses["wait"]):
            edge(2 * release + 1, 2 * acquire + 1, "matching producer completion and consumer acquire")

    reachable = {}
    active = set()

    def visit(vertex):
        if vertex in active:
            raise ValueError("The token protocol contains an execution cycle")
        if vertex not in reachable:
            active.add(vertex)
            result = set()
            for target in adjacency[vertex]:
                result.add(target)
                result.update(visit(target))
            active.remove(vertex)
            reachable[vertex] = result
        return reachable[vertex]

    cycle = None
    try:
        for vertex in range(2 * len(events)):
            visit(vertex)
    except ValueError as error:
        cycle = str(error)

    def before(source, target):
        return cycle is None and target in reachable[source]

    for channel, uses in channels.items():
        for acquire, next_release in zip(uses["wait"], uses["set"][1:]):
            if not before(2 * acquire + 1, 2 * next_release):
                token_errors.append(
                    {
                        "channel": channel,
                        "error": "binary token can be released again before acquire",
                        "acquire": acquire,
                        "release": next_release,
                    }
                )

    generations = {"a": [], "b": []}
    current = {}
    read_errors = []
    reader_counts = defaultdict(int)
    for index, event in enumerate(events):
        for access in event.get("reads", []):
            storage = access["storage"]
            if storage not in generations:
                continue
            reader_counts[storage + ":" + event["pipe"]] += 1
            cell = (storage, tuple(access["index"]))
            if cell not in current:
                read_errors.append({"event": index, "error": "read has no preceding definition", "access": access})
                continue
            generation = current[cell]
            generation["reads"].append(index)
            generation["ends"].append(2 * index + 1)
            if not before(2 * generation["writer"] + 1, 2 * index):
                read_errors.append(
                    {"event": index, "error": "definition does not complete before reader issues", "writer": generation["writer"]}
                )
        for access in event.get("writes", []):
            storage = access["storage"]
            if storage not in generations:
                continue
            cell = (storage, tuple(access["index"]))
            generation = {
                "storage": storage,
                "cell": access["index"],
                "writer": index,
                "reads": [],
                "begin": 2 * index,
                "ends": [2 * index + 1],
            }
            current[cell] = generation
            generations[storage].append(generation)

    overlaps = []
    separated = []
    for left in generations["a"]:
        for right in generations["b"]:
            left_first = all(before(end, right["begin"]) for end in left["ends"])
            right_first = all(before(end, left["begin"]) for end in right["ends"])
            if not left_first and not right_first:
                overlaps.append(
                    {
                        "a_writer": left["writer"],
                        "b_writer": right["writer"],
                        "a_readers": left["reads"],
                        "b_readers": right["reads"],
                        "a_ends_not_before_b": [end // 2 for end in left["ends"] if not before(end, right["begin"])],
                        "b_ends_not_before_a": [end // 2 for end in right["ends"] if not before(end, left["begin"])],
                    }
                )
            else:
                separated.append([left["writer"], right["writer"], "a-before-b" if left_first else "b-before-a"])

    drains = [event["id"] for event in events if event["kind"] == "wait" and event["channel"] == "MTE3_S" and not event["loops"]]
    seeds = [event["id"] for event in events if event["kind"] == "set" and event["channel"] == "MTE3_S" and not event["loops"]]
    dma = [event["id"] for event in events if event["kind"] == "copy"]
    drained = bool(drains) and all(any(before(2 * transfer + 1, 2 * drain + 1) for drain in drains) for transfer in dma)
    seeded = bool(seeds) and all(events[seed]["id"] < min(dma) for seed in seeds)
    return {
        "safe": bool(generations["a"] and generations["b"] and dma and seeded and drained)
        and not (token_errors or read_errors or overlaps or cycle),
        "events": events,
        "edges": edge_records,
        "token_channels": [{"key": list(key), **uses} for key, uses in channels.items()],
        "token_errors": token_errors,
        "read_errors": read_errors,
        "cycle": cycle,
        "reader_counts": dict(reader_counts),
        "generations": generations,
        "separated_pairs": separated,
        "unordered_pairs": overlaps,
        "seed_events": seeds,
        "drain_events": drains,
        "all_dma_drained": drained,
        "seeded_before_first_dma": seeded,
    }
