"""Smoke test for the experimental CPython 3.13–3.14 VM hooks.

Run with an interpreter built from a patch in native/patches.
"""

import dis
import sys


events = []
mutations = []
function_events = []
global_value = 0
mapping = {}
sequence = [0, 0, 0]


class Box:
    pass


class SlottedBox:
    __slots__ = ("value",)


box = Box()
slotted_box = SlottedBox()


def before_store(frame, name, value):
    internal = "the recursion guard must hide this"
    if frame.f_code.co_filename in {__file__, "<string>"}:
        events.append((frame.f_code.co_name, name, value))
    if name == "blocked":
        raise TypeError("blocked by the store hook")


def before_mutation(frame, kind, owner, key, value):
    if frame.f_code.co_filename == __file__:
        mutations.append((kind, owner, key, value))
    # An assignment made by a callback must not recursively invoke the hook.
    owner_guard = Box()
    owner_guard.internal = 1
    if key == "blocked":
        raise TypeError("blocked by the mutation hook")


def on_call(frame):
    if frame.f_code.co_name == "rejected_entry":
        raise TypeError("entry rejected")
    if frame.f_code.co_name == "checked_function":
        function_events.append(("call", frame.f_locals["value"]))


def on_return(frame, value):
    if frame.f_code.co_name == "rejected_return":
        raise TypeError("return rejected")
    if frame.f_code.co_name == "checked_function":
        function_events.append(("return", value))


def local_store():
    first = 1
    second = 2
    try:
        blocked = 3
    except TypeError:
        return first, second
    raise AssertionError("store hook did not reject the assignment")


def global_store():
    global global_value
    global_value = 4


def closure_store():
    value = 5

    def inner():
        nonlocal value
        value = 6

    inner()
    return value


def specialized_mutations(value):
    box.value = value
    slotted_box.value = value
    mapping["value"] = value
    sequence[0] = value


def rejected_mutations():
    try:
        box.blocked = 1
    except TypeError:
        assert not hasattr(box, "blocked")
    else:
        raise AssertionError("attribute store was not rejected")
    try:
        mapping["blocked"] = 2
    except TypeError:
        assert "blocked" not in mapping
    else:
        raise AssertionError("item store was not rejected")


def slice_mutation():
    sequence[1:3] = [7, 8]


def checked_function(value):
    return value + 1


def rejected_entry():
    raise AssertionError("entry hook should prevent execution")


def rejected_return():
    return 9


def tracer(frame, event, argument):
    return tracer


def unicode_inplace():
    text = "a"
    text += "b"
    return text


def main():
    for _ in range(1000):
        assert unicode_inplace() == "ab"
    sys._aiython_before_store = before_store
    try:
        assert unicode_inplace() == "ab"
        assert local_store() == (1, 2)
        global_store()
        assert global_value == 4
        assert closure_store() == 6
        namespace = {}
        exec("module_value = 7\nclass Example:\n    class_value = 8", namespace)
        assert namespace["module_value"] == 7
        assert namespace["Example"].class_value == 8
        try:
            exec("blocked = 9", namespace)
        except TypeError:
            assert "blocked" not in namespace
        else:
            raise AssertionError("module store did not reject the assignment")
        assert list(dis.get_instructions(local_store))
        for value in range(1000):
            specialized_mutations(value)
        names = {instruction.opname for instruction in
                 dis.get_instructions(specialized_mutations, adaptive=True)}
        assert "STORE_ATTR_INSTANCE_VALUE" in names, names
        assert "STORE_ATTR_SLOT" in names, names
        assert "STORE_SUBSCR_DICT" in names, names
        assert "STORE_SUBSCR_LIST_INT" in names, names
        sys._aiython_before_mutation = before_mutation
        specialized_mutations(1001)
        rejected_mutations()
        slice_mutation()
        assert box.value == slotted_box.value == mapping["value"] == sequence[0] == 1001
        assert sequence[1:3] == [7, 8]
        sys._aiython_on_call = on_call
        sys._aiython_on_return = on_return
        assert checked_function(3) == 4
        sys.settrace(tracer)
        try:
            assert checked_function(10) == 11
        finally:
            sys.settrace(None)
        try:
            rejected_entry()
        except TypeError as error:
            assert str(error) == "entry rejected"
        else:
            raise AssertionError("entry hook did not reject the call")
        try:
            rejected_return()
        except TypeError as error:
            assert str(error) == "return rejected"
        else:
            raise AssertionError("return hook did not reject the value")
    finally:
        del sys._aiython_before_store
        if hasattr(sys, "_aiython_before_mutation"):
            del sys._aiython_before_mutation
        if hasattr(sys, "_aiython_on_call"):
            del sys._aiython_on_call
        if hasattr(sys, "_aiython_on_return"):
            del sys._aiython_on_return

    expected = {
        ("local_store", "first", 1),
        ("local_store", "second", 2),
        ("local_store", "blocked", 3),
        ("global_store", "global_value", 4),
        ("inner", "value", 6),
        ("<module>", "module_value", 7),
        ("Example", "class_value", 8),
        ("<module>", "blocked", 9),
        ("unicode_inplace", "text", "ab"),
    }
    missing = [item for item in expected if item not in events]
    assert not missing, missing
    assert not any(name == "internal" for _, name, _ in events)
    assert any(kind == "attr" and owner is box and key == "value" and value == 1001
               for kind, owner, key, value in mutations)
    assert any(kind == "attr" and owner is slotted_box and key == "value" and value == 1001
               for kind, owner, key, value in mutations)
    assert any(kind == "item" and owner is mapping and key == "value" and value == 1001
               for kind, owner, key, value in mutations)
    assert any(kind == "item" and owner is sequence and key == 0 and value == 1001
               for kind, owner, key, value in mutations)
    assert any(kind == "item" and owner is sequence and isinstance(key, slice)
               and key == slice(1, 3) for kind, owner, key, value in mutations)
    assert not any(key == "internal" for _, _, key, _ in mutations)
    assert function_events == [
        ("call", 3), ("return", 4), ("call", 10), ("return", 11)
    ], function_events
    version = ".".join(map(str, sys.version_info[:3]))
    print(f"CPython {version} VM hook smoke passed")


if __name__ == "__main__":
    main()
