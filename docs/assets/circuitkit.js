/* Make the header title navigate home, matching the logo's behavior. */
document$.subscribe(function () {
  var title = document.querySelector(".md-header__title");
  if (title && !title.dataset.ckHome) {
    title.dataset.ckHome = "1";
    title.style.cursor = "pointer";
    title.addEventListener("click", function () {
      var logo = document.querySelector(".md-header__button.md-logo");
      if (logo && logo.href) {
        window.location.href = logo.href;
      }
    });
  }
});

/* Landing-hero: a layered network in which discovered circuits resolve, one hop
   at a time, from idle to the brand accent — then release and repeat. */
(function () {
  var NS = "http://www.w3.org/2000/svg";
  var HOLD = 1600, GAP = 620, STAGGER = 240, PULSE_MS = 700;
  var teardown = null;

  function svgEl(name, attrs) {
    var e = document.createElementNS(NS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }

  function pointAlong(pts, u) {
    var n = pts.length - 1;
    var s = Math.max(0, Math.min(n - 1, Math.floor(u * n)));
    var f = u * n - s;
    return [
      pts[s][0] + (pts[s + 1][0] - pts[s][0]) * f,
      pts[s][1] + (pts[s + 1][1] - pts[s][1]) * f
    ];
  }

  function initHeroCircuit(host) {
    var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var timers = new Set();
    var alive = true;

    function later(fn, ms) {
      var id = window.setTimeout(function () { timers.delete(id); fn(); }, ms);
      timers.add(id);
    }

    var svg = svgEl("svg", { viewBox: "0 0 440 260", "aria-hidden": "true", focusable: "false" });
    var gEdges = svgEl("g", {}), gNodes = svgEl("g", {});
    svg.appendChild(gEdges);
    svg.appendChild(gNodes);

    var layers = [
      { x: 58, ys: [74, 146, 218] },
      { x: 178, ys: [54, 118, 182, 238] },
      { x: 298, ys: [54, 118, 182, 238] },
      { x: 392, ys: [74, 146, 218] }
    ];
    var nodeEls = [], idx = [];
    layers.forEach(function (L, li) {
      idx[li] = [];
      L.ys.forEach(function (y) {
        var c = svgEl("circle", { "class": "ck-node", cx: L.x, cy: y, r: 7.5 });
        gNodes.appendChild(c);
        idx[li].push(nodeEls.length);
        nodeEls.push(c);
      });
    });

    var edges = [], emap = {};
    for (var li = 0; li < layers.length - 1; li++) {
      layers[li].ys.forEach(function (ya, a) {
        layers[li + 1].ys.forEach(function (yb, b) {
          var x1 = layers[li].x, x2 = layers[li + 1].x;
          var line = svgEl("line", { "class": "ck-edge", x1: x1, y1: ya, x2: x2, y2: yb });
          gEdges.appendChild(line);
          emap[li + ":" + a + ":" + b] = edges.length;
          edges.push({ el: line, na: idx[li][a], nb: idx[li + 1][b], points: [[x1, ya], [x2, yb]] });
        });
      });
    }

    var circuits = [[1, 2, 1, 1], [0, 0, 3, 0], [2, 3, 2, 2], [0, 3, 1, 2]].map(function (p) {
      var eds = [];
      for (var i = 0; i < layers.length - 1; i++) eds.push(emap[i + ":" + p[i] + ":" + p[i + 1]]);
      return eds;
    });

    host.textContent = "";
    host.appendChild(svg);

    var onEls = [];

    function pulse(points) {
      if (reduce) return;
      var dot = svgEl("circle", { "class": "ck-pulse", r: 3.4, cx: points[0][0], cy: points[0][1] });
      svg.appendChild(dot);
      var t0 = performance.now();
      function frame(t) {
        if (!alive) { if (dot.parentNode) dot.remove(); return; }
        var u = Math.max(0, Math.min(1, (t - t0) / PULSE_MS));
        var q = pointAlong(points, u);
        dot.setAttribute("cx", q[0]);
        dot.setAttribute("cy", q[1]);
        if (u < 1) window.requestAnimationFrame(frame);
        else dot.remove();
      }
      window.requestAnimationFrame(frame);
    }

    function activate(circuit) {
      circuit.forEach(function (ei, k) {
        later(function () {
          if (!alive) return;
          var e = edges[ei];
          e.el.classList.add("is-on");
          var a = nodeEls[e.na], b = nodeEls[e.nb];
          a.classList.add("is-on");
          b.classList.add("is-on");
          onEls.push(e.el, a, b);
          pulse(e.points);
        }, k * STAGGER);
      });
    }

    function deactivate() {
      onEls.forEach(function (n) { n.classList.remove("is-on"); });
      onEls = [];
    }

    if (reduce) {
      activate(circuits[0]);
    } else {
      var i = 0;
      (function step() {
        if (!alive) return;
        var c = circuits[i % circuits.length];
        activate(c);
        later(function () {
          if (!alive) return;
          deactivate();
          later(step, GAP);
        }, HOLD + c.length * STAGGER);
        i++;
      })();
    }

    return function () {
      alive = false;
      timers.forEach(function (id) { window.clearTimeout(id); });
      timers.clear();
    };
  }

  function boot() {
    if (teardown) { teardown(); teardown = null; }
    var host = document.getElementById("ck-hero-anim");
    if (host) teardown = initHeroCircuit(host);
  }

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(boot);
  } else {
    document.addEventListener("DOMContentLoaded", boot);
  }
})();
