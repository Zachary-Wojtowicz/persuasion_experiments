// ---------------------------------------------------------------------------
// Qualtrics JavaScript: populate the "essay" embedded data field.
//
// Paste this into a question's "Add JavaScript" editor (onload section).
// Requires three embedded data fields set earlier in the survey flow:
//   - topic    ∈ {"guns", "abortion"}
//   - politics ∈ {"liberal", "conservative"}
//   - tone     ∈ {"strong", "weak"}
//
// To generate the ESSAYS data blob, run in Python:
//
//   import json
//   with open("all_essays.json") as f:
//       data = json.load(f)
//   lookup = {}
//   for topic_obj in data:
//       issue = topic_obj["issue"]
//       lookup[issue] = {}
//       for sr in topic_obj["stance_runs"]:
//           pol = sr["politics"]
//           lookup[issue][pol] = {}
//           for e in sr["essays"]:
//               lookup[issue][pol].setdefault(e["tone"], []).append(e["essay"])
//               # (setdefault is Python; the JS equivalent is built below)
//   print(json.dumps(lookup, indent=2))
//
// Then paste the output as the value of ESSAYS below.
// ---------------------------------------------------------------------------

Qualtrics.SurveyEngine.addOnload(function () {

    // ---- BEGIN ESSAY DATA (paste generated JSON here) ----
    var ESSAYS = {
        "guns": {
            "liberal": {
                "strong": [
                    // essay texts ...
                ],
                "weak": [
                    // essay texts ...
                ]
            },
            "conservative": {
                "strong": [],
                "weak": []
            }
        },
        "abortion": {
            "liberal": {
                "strong": [],
                "weak": []
            },
            "conservative": {
                "strong": [],
                "weak": []
            }
        }
    };
    // ---- END ESSAY DATA ----

    // Read the randomized embedded data fields (piped text resolved server-side)
    var topic    = "${e://Field/topic}";
    var politics = "${e://Field/politics}";
    var tone     = "${e://Field/tone}";

    // Look up the matching essay pool and pick one at random
    var pool = ESSAYS[topic] && ESSAYS[topic][politics] && ESSAYS[topic][politics][tone];

    if (pool && pool.length > 0) {
        var idx   = Math.floor(Math.random() * pool.length);
        var essay = pool[idx];
        Qualtrics.SurveyEngine.setEmbeddedData("essay", essay);
    } else {
        // Fallback: log a warning so you can diagnose in preview
        console.warn(
            "No essay found for topic=" + topic +
            ", politics=" + politics +
            ", tone=" + tone
        );
    }

});
