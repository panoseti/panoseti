#! /usr/bin/env python3
"""Methods for generating aggregate instrument health logs for observatory operators.
 Log messages are added to the appropriate metadata field and are displayed on the Grafana webpage."""
import json

with open("metadata_status_monitor_config.json", "r") as f:
    status_states = json.load(f)

status_map = {
    "ok": 0,
    "warn": 1,
    "critical": 2
}

status_history = dict()


def write_status(datatype, redis_key, metadata_dict):
    """
    Get the current status message and level (0,1,2) for redis_key. Then, write into metadata_dict either:
        1. The status message and level if status has changed since last update.
        2. "" for status message and -1 for status level if the status has not changed since the last update.
    The purpose of 2 is to save memory by reducing redundant log messages.
    """
    status = get_status("housekeeping", metadata_dict)
    new_status = (redis_key not in status_history) or (status_history[redis_key] != status)
    if new_status:
        status_history[redis_key] = status
        metadata_dict['AGG_STATUS_MSG'] = status[0]
        metadata_dict['AGG_STATUS_LEVEL'] = status[1]
    else:
        metadata_dict['AGG_STATUS_MSG'] = ""
        metadata_dict['AGG_STATUS_LEVEL'] = -1


def get_status(datatype, metadata_dict):
    """
    This creates a log message for the Grafana webpage to report warnings or more serious issues
    an operator should address while monitoring an observing run.

    datatype is either 'housekeeping' for quabos, 'GPS',
    'whiterabbit', or 'outlet' for WPS.

    metadata_dict is a dictionary of metadata created by one of the metadata capture scripts
    before it gets entered into redis.
    """
    status_msg = ""
    status_level = -1
    for entry in status_states[datatype]:
        if len(entry["fields"]) == 0:
            continue
        name = entry["name"]
        metadata_values = []
        for field in entry["fields"]:
            metadata_values.append(metadata_dict[field])
        in_this_state = False
        for state in entry["states"]:
            status = state["status"]
            message = state["message"]
            for condition in state["condition"]:
                if state["condition"] == "else":
                    in_this_state = True
                    #print(f'{name}, else')
                    break
                for val in metadata_values:
                    in_this_state |= condition[0] <= val < condition[1]
                    #print(f'{name}, {condition[0]}<={val}<{condition[1]} == {condition[0] <= val < condition[1]}')
            if in_this_state:
                if status != "ok":
                    status_msg += f"<<{name}:{status}:'{message}'>>"
                status_level = max(status_level, status_map[status])
                break
    return status_msg, status_level


"""
test = {
    "TEMP2": 4.99,
    "TEMP1": -10.1,
    "HVMON0": -14.9,
    "HVMON1": 20,
    "HVMON2": 20,
    "HVMON3": 20,
}

write_status("housekeeping", "TEST", test)
print(test['AGG_STATUS_LEVEL'])
write_status("housekeeping", "TEST", test)
print(test['AGG_STATUS_LEVEL'])
test["TEMP1"] = 87
write_status("housekeeping", "TEST", test)
print(test['AGG_STATUS_LEVEL'])
write_status("housekeeping", "TEST", test)
print(test['AGG_STATUS_LEVEL'])
write_status("housekeeping", "TEST", test)
print(test['AGG_STATUS_LEVEL'])
write_status("housekeeping", "TEST", test)
print(test['AGG_STATUS_LEVEL'])
test["TEMP2"] = 100
write_status("housekeeping", "TEST", test)
print(test['AGG_STATUS_LEVEL'])
write_status("housekeeping", "TEST", test)
print(test['AGG_STATUS_LEVEL'])
test["TEMP2"] = 15
test["TEMP1"] = 17
write_status("housekeeping", "TEST", test)
print(test['AGG_STATUS_LEVEL'])
test["HVMON0"] = 20
write_status("housekeeping", "TEST", test)
print(test['AGG_STATUS_LEVEL'])
write_status("housekeeping", "TEST", test)
print(test['AGG_STATUS_LEVEL'])
"""
