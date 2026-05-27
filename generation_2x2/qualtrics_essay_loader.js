Qualtrics.SurveyEngine.addOnReady(function () {
    // 2x2 design: confidence x brashness. Update this URL to wherever the
    // generated JSON is served (e.g. GitHub Pages, S3). The file should be the
    // direct output of generation_2x2/generate_arguments.py.
    var URL = "https://Zachary-Wojtowicz.github.io/persuasion_experiments/generation_2x2/essays_final_2x2.json";

    // Embedded data fields (Qualtrics resolves these server-side before this runs).
    // Compared with the prior single-axis design, the field "tone" has been
    // replaced by two fields: "confidence" and "brashness". Make sure the
    // survey's randomizer assigns both before this loader runs.
    var topic      = "${e://Field/topic}";
    var politics   = "${e://Field/politics}";
    var confidence = "${e://Field/confidence}"; // expects "low" or "high"
    var brashness  = "${e://Field/brashness}";  // expects "low" or "high"

    // If an essay has already been assigned earlier in the session, don't
    // reassign on refresh/back. (Delete this block to force reassignment.)
    var existingEssay = "${e://Field/essay}";
    if (existingEssay && existingEssay !== "" && existingEssay !== "null") {
      return;
    }

    function norm(x) {
      return (x || "").toString().trim().toLowerCase();
    }

    function pickRandom(arr) {
      return arr[Math.floor(Math.random() * arr.length)];
    }

    // Wrap async in an IIFE because Qualtrics addOnReady isn't declared async
    (async function () {
      try {
        var res = await fetch(URL, { cache: "no-store" });
        if (!res.ok) throw new Error("Fetch failed: HTTP " + res.status);

        var data = await res.json();

        // Support both formats: {config, results} wrapper or bare array
        var results = Array.isArray(data) ? data : (data.results || []);
        if (!results.length) {
          throw new Error("Unexpected JSON format: no results found.");
        }

        var tTopic = norm(topic);
        var tConf  = norm(confidence);
        var tBrash = norm(brashness);

        // Flip politics so participants see counter-attitudinal essays:
        // essays tagged "liberal" were written to persuade a conservative
        // audience, and vice versa.
        var oppositeMap = { "liberal": "conservative", "conservative": "liberal" };
        var tPol = oppositeMap[norm(politics)] || norm(politics);

        // Walk results -> stance_runs -> quads -> essays, filtering on
        // topic + (flipped) politics + confidence + brashness.
        var candidates = [];

        for (var i = 0; i < results.length; i++) {
          var issueObj = results[i];
          if (norm(issueObj.issue) !== tTopic) continue;

          var stanceRuns = issueObj.stance_runs || [];
          for (var sr = 0; sr < stanceRuns.length; sr++) {
            var run = stanceRuns[sr];
            if (norm(run.politics) !== tPol) continue;

            var quads = run.quads || [];
            for (var q = 0; q < quads.length; q++) {
              var quad = quads[q];
              var essays = quad.essays || [];
              for (var e = 0; e < essays.length; e++) {
                var es = essays[e];
                if (norm(es.confidence) !== tConf) continue;
                if (norm(es.brashness)  !== tBrash) continue;

                candidates.push({
                  issue: issueObj.issue,
                  politics: run.politics,
                  confidence: es.confidence,
                  brashness:  es.brashness,
                  quad_index: quad.quad_index,
                  stance: run.stance,
                  contrary_stance: run.contrary_stance,
                  essay: es.essay
                });
              }
            }
          }
        }

        if (candidates.length === 0) {
          console.warn(
            "No essay found for topic=" + topic +
            ", politics=" + politics +
            ", confidence=" + confidence +
            ", brashness=" + brashness
          );
          // You can optionally set a fallback embedded value:
          // Qualtrics.SurveyEngine.setEmbeddedData("essay", "MISSING_ESSAY");
          return;
        }

        var chosen = pickRandom(candidates);

        // Save the essay text with newlines converted to <br> for HTML display
        Qualtrics.SurveyEngine.setEmbeddedData("essay", chosen.essay.replace(/\n/g, "<br>"));

        // Save metadata for analysis/debugging
        Qualtrics.SurveyEngine.setEmbeddedData("essay_issue", chosen.issue);
        Qualtrics.SurveyEngine.setEmbeddedData("essay_politics", chosen.politics);
        Qualtrics.SurveyEngine.setEmbeddedData("essay_confidence", chosen.confidence);
        Qualtrics.SurveyEngine.setEmbeddedData("essay_brashness", chosen.brashness);
        Qualtrics.SurveyEngine.setEmbeddedData("essay_quad_index", chosen.quad_index);
        Qualtrics.SurveyEngine.setEmbeddedData("essay_stance", chosen.stance);
        Qualtrics.SurveyEngine.setEmbeddedData("essay_contrary_stance", chosen.contrary_stance);

      } catch (err) {
        console.error("Error loading/selecting essay:", err);
      }
    })();
  });
