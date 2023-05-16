#! /usr/bin/env python3
"""Methods for generating aggregate instrument health logs for observatory operators.
 Log messages are added to the appropriate metadata field and are displayed on the Grafana webpage."""
import json

with open("metadata_status_monitor_config.json", "r") as f:
    status_states = json.load(f)

status_map = {
    "ok": 0,
    "info": 8,
    "warn": 16,
    "critical": 32
}

status_history = dict()


def write_status(datatype, redis_key, metadata_dict):
    """
    Get the current status redis_key then write it into metadata_dict.
    """
    status = get_status(datatype, metadata_dict)
    new_status = (redis_key not in status_history) or (status_history[redis_key] != status)
    status_history[redis_key] = status
    metadata_dict['AGG_STATUS_MSG'] = status[0]
    metadata_dict['AGG_STATUS_LEVEL'] = status[1]


def get_status(datatype, metadata_dict):
    """
    This creates a log message for the Grafana webpage to report statuses
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
                    status_msg += f"{name},"
                status_level = max(status_level, status_map[status])
                break
    if len(status_msg) == 0:
        status_msg = "ok,"
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
