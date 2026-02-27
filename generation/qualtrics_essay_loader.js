Qualtrics.SurveyEngine.addOnReady(function () {
    var URL = "https://Zachary-Wojtowicz.github.io/persuasion_experiments/generation/all_essays.json";
  
    // Read the randomized embedded data fields (piped text resolved server-side)
    var topic    = "${e://Field/topic}";
    var politics = "${e://Field/politics}";
    var tone     = "${e://Field/tone}"; // expects "strong" or "weak"
  
    // Optional: if you've already assigned an essay earlier, don't reassign on refresh/back.
    // (If you *want* reassignment on refresh, delete this block.)
    var existingEssay = "${e://Field/essay}";
    if (existingEssay && existingEssay !== "" && existingEssay !== "null") {
      return;
    }
  
    // Helper: normalize to lower-case just in case your embedded data varies in casing
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
  
        if (!Array.isArray(data)) {
          throw new Error("Unexpected JSON format: expected top-level array.");
        }
  
        var tTopic = norm(topic);
        var tPol   = norm(politics);
        var tTone  = norm(tone);
  
        // Collect all matching essays across all stance_runs + pairs
        var candidates = [];
  
        for (var i = 0; i < data.length; i++) {
          var issueObj = data[i];
          if (norm(issueObj.issue) !== tTopic) continue;
  
          var stanceRuns = issueObj.stance_runs || [];
          for (var sr = 0; sr < stanceRuns.length; sr++) {
            var run = stanceRuns[sr];
            if (norm(run.politics) !== tPol) continue;
  
            var pairs = run.pairs || [];
            for (var p = 0; p < pairs.length; p++) {
              var pair = pairs[p];
              var essays = pair.essays || [];
              for (var e = 0; e < essays.length; e++) {
                var es = essays[e];
                if (norm(es.tone) !== tTone) continue;
  
                candidates.push({
                  issue: issueObj.issue,
                  politics: run.politics,
                  tone: es.tone,
                  pair_index: pair.pair_index,
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
            ", tone=" + tone
          );
          // You can optionally set a fallback embedded value:
          // Qualtrics.SurveyEngine.setEmbeddedData("essay", "MISSING_ESSAY");
          return;
        }
  
        var chosen = pickRandom(candidates);
  
        // Save the essay text with newlines converted to <br> for HTML display
        Qualtrics.SurveyEngine.setEmbeddedData("essay", chosen.essay.replace(/\n/g, "<br>"));
  
        // Optional: save metadata for analysis/debugging
        Qualtrics.SurveyEngine.setEmbeddedData("essay_issue", chosen.issue);
        Qualtrics.SurveyEngine.setEmbeddedData("essay_politics", chosen.politics);
        Qualtrics.SurveyEngine.setEmbeddedData("essay_tone", chosen.tone);
        Qualtrics.SurveyEngine.setEmbeddedData("essay_pair_index", chosen.pair_index);
        Qualtrics.SurveyEngine.setEmbeddedData("essay_stance", chosen.stance);
        Qualtrics.SurveyEngine.setEmbeddedData("essay_contrary_stance", chosen.contrary_stance);
  
      } catch (err) {
        console.error("Error loading/selecting essay:", err);
      }
    })();
  });