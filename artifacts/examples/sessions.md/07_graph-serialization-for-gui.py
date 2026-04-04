# Source   : lazy_wiki/human/sessions.md
# Heading  : Graph serialization (for GUI)
# ID       : lazy_wiki/human/sessions.md::graph-serialization-for-gui::00
# Kind     : local
# Testable : local_exec

import json

# Export as JSON (for a GUI to load)
json_str = sess.to_json()
print(json_str)

# Save to file
sess.graph.save("pipeline.json")
sess.graph.save("pipeline.yaml")   # requires pip install pyyaml
