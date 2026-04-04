# Source   : lazy_wiki/human/sessions.md
# Heading  : Event tracking
# ID       : lazy_wiki/human/sessions.md::event-tracking::00
# Kind     : local
# Testable : local_exec

from lazybridge import Event

# All events
all_events = sess.events.get()

# Filter by event type
tool_calls = sess.events.get(event_type=Event.TOOL_CALL)
responses  = sess.events.get(event_type=Event.MODEL_RESPONSE)

# Filter by agent
writer_events = sess.events.get(agent_id=writer.id)

# Both filters + limit
recent = sess.events.get(agent_id=researcher.id, event_type=Event.TOOL_RESULT, limit=10)

# Each event is a dict:
for ev in all_events:
    print(ev["timestamp"], ev["agent_name"], ev["event_type"], ev["data"])
