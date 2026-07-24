// Re-scroll to the URL fragment once the page has finished laying out. Sphinx renders LaTeX with MathJax
// asynchronously: the browser jumps to a deep anchor immediately, then MathJax typesets the formulas above it, which
// grows the content and pushes the target down — leaving the reader parked on an earlier function. Re-applying the
// hash after MathJax settles fixes cross-page anchor links (e.g. a benchmark chart label into the API reference).
(function () {
  "use strict";

  function reanchor() {
    if (!location.hash) return;
    var target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
    if (target) target.scrollIntoView();
  }

  window.addEventListener("load", function () {
    if (window.MathJax && window.MathJax.startup && window.MathJax.startup.promise) {
      window.MathJax.startup.promise.then(reanchor);
    } else {
      setTimeout(reanchor, 250);
    }
  });
})();
