# penaltyblog/matchflow/opta_helpers.py
from typing import Any, List, Optional, Union

from .opta_mappings import OPTA_EVENT_DEFINITIONS, OPTA_QUALIFIER_DEFINITIONS
from .predicates_helpers import where_equals, where_in

# Reverse mapping for quick lookup (Event Name -> typeId)
# Handle potential case differences and ensure names are unique if possible
OPTA_EVENT_NAME_TO_ID = {
    details["name"].lower(): type_id
    for type_id, details in OPTA_EVENT_DEFINITIONS.items()
}

OPTA_QUALIFIER_NAME_TO_ID = {
    details["name"].lower(): q_id
    for q_id, details in OPTA_QUALIFIER_DEFINITIONS.items()
}


def where_opta_event(event_name: Union[str, List[str]]):
    """
    Creates a predicate to filter Opta events by their type NAME(s).

    Looks up the corresponding typeId(s) from the Opta mappings.

    Args:
        event_name: The name of the event type (e.g., "Pass", "Shot")
                    or a list of names. Case-insensitive.

    Returns:
        A matchflow predicate function.

    Raises:
        ValueError: If an event name is not found in the mappings.
    """
    if isinstance(event_name, str):
        # Single event name
        name_lower = event_name.lower()
        type_id = OPTA_EVENT_NAME_TO_ID.get(name_lower)
        if type_id is None:
            # Maybe try replacing spaces with underscores or vice-versa if needed?
            # Example: name_lower = name_lower.replace(" ", "_")
            # type_id = OPTA_EVENT_NAME_TO_ID.get(name_lower)
            if type_id is None:
                raise ValueError(f"Unknown Opta event type name: '{event_name}'")
        # Use where_equals for single ID
        return where_equals("typeId", type_id)
    elif isinstance(event_name, list):
        # List of event names
        type_ids = []
        for name in event_name:
            name_lower = name.lower()
            type_id = OPTA_EVENT_NAME_TO_ID.get(name_lower)
            if type_id is None:
                # Add same fallback logic as above if necessary
                raise ValueError(f"Unknown Opta event type name in list: '{name}'")
            type_ids.append(type_id)
        # Use where_in for multiple IDs
        return where_in("typeId", type_ids)
    else:
        raise TypeError("event_name must be a string or a list of strings")


def where_opta_qualifier(qualifier_name: str, value: Optional[Any] = None):
    """
    Creates a predicate to filter Opta events based on the presence
    or specific value of a qualifier NAME.

    Looks up the corresponding qualifierId from the Opta mappings.

    Args:
        qualifier_name: The name of the qualifier (e.g., "Penalty", "Zone").
                        Case-insensitive.
        value: The specific value the qualifier should have (e.g., "Back").
               If None (default), checks only for the qualifier's presence.

    Returns:
        A function that takes an event dictionary and returns True/False.

    Raises:
        ValueError: If the qualifier name is not found in the mappings.
    """
    q_name_lower = qualifier_name.lower()
    qualifier_id = OPTA_QUALIFIER_NAME_TO_ID.get(q_name_lower)

    if qualifier_id is None:
        raise ValueError(f"Unknown Opta qualifier name: '{qualifier_name}'")

    def predicate(event: dict) -> bool:
        qualifiers = event.get("qualifier", [])
        if not isinstance(
            qualifiers, list
        ):  # Handle cases where it might not be a list
            return False

        for q in qualifiers:
            if q.get("qualifierId") == qualifier_id:
                # Found the qualifier by ID
                if value is None:
                    return True  # Presence check successful
                else:
                    # Value check: Use simple equality for now.
                    # May need refinement if type comparison becomes an issue (e.g., '1' vs 1)
                    if q.get("value") == value:
                        return True  # Value matches
        # If loop finishes without finding a match
        return False

    return predicate


def get_opta_mappings() -> dict:
    """
    Returns the available Opta event and qualifier names and their IDs.

    Returns:
        dict: {
            "events": List[dict],  # Each dict: {"id": type_id, "name": name}
            "qualifiers": List[dict],  # Each dict: {"id": q_id, "name": name}
        }
    """
    events = [
        {"id": type_id, "name": details["name"]}
        for type_id, details in sorted(
            OPTA_EVENT_DEFINITIONS.items(), key=lambda item: item[1]["name"]
        )
    ]
    qualifiers = [
        {"id": q_id, "name": details["name"]}
        for q_id, details in sorted(
            OPTA_QUALIFIER_DEFINITIONS.items(), key=lambda item: item[1]["name"]
        )
    ]
    return {
        "events": events,
        "qualifiers": qualifiers,
    }
