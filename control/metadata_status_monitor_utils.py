#! /usr/bin/env python3
"""Methods for generating aggregate instrument health logs for observatory operators.
 Log messages are added to the appropriate metadata field and are displayed on the Grafana webpage."""
import json

with open("metadata_status_monitor_config.json", "r") as f:
    status_states = json.load(f)

status_history = dict()


def get_status(datatype, redis_key, metadata_dict):
    """
    Get the current status message for redis_key.
    Returns the current status message if it has changed since its last update.
    """
    status_msg = generate_status_msg(datatype, metadata_dict)
    if (redis_key not in status_history) or (status_history[redis_key] != status_msg):
        status_history[redis_key] = status_msg
        return status_msg
    return status_msg


def generate_status_msg(datatype, metadata_dict):
    """
    This creates a log message for the Grafana webpage to report warnings or more serious issues
    an operator should address while monitoring an observing run.

    datatype is either 'housekeeping' for quabos, 'GPS',
    'whiterabbit', or 'outlet' for WPS.

    metadata_dict is a dictionary of metadata created by one of the metadata capture scripts
    before it gets entered into redis.
    """
    status_msg = ""
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
                    break
                for val in metadata_values:
                    in_this_state |= condition[0] <= val < condition[1]
                    #print(f'{name}, {condition[0]}<={val}<{condition[1]} == {condition[0] <= val < condition[1]}')
            if in_this_state:
                if status != "ok":
                    status_msg += f"<<{name}:{status}:'{message}'>>"
                break
    #print(status_msg)
    return status_msg


test = {
    "TEMP2": 4.99,
    "TEMP1": -10.1,
    "HVMON0": -14.9,
    "HVMON1": 20,
    "HVMON2": 20,
    "HVMON3": 20,
}

'''
print(get_status("housekeeping", "TEST", test))
print(get_status("housekeeping", "TEST", test))
test["TEMP1"] = 5
print(get_status("housekeeping", "TEST", test))
print(get_status("housekeeping", "TEST", test))
'''