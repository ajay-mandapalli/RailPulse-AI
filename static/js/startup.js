/* Rendered by Flask once, only after successful authentication. */
(() => {
    "use strict";
    const splash = document.getElementById("rp-startup");
    if (!splash) return;

    const root = document.documentElement;
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    let fallback;
    let finished = false;
    const covered = new Map();

    function protectPage() {
        if (finished) return;
        for (const sibling of document.body.children) {
            if (sibling === splash || ["SCRIPT", "STYLE", "LINK"].includes(sibling.tagName)) continue;
            covered.set(sibling, sibling.inert);
            sibling.inert = true;
        }
    }

    function preventScroll(event) {
        event.preventDefault();
    }

    function finish() {
        if (finished) return;
        finished = true;
        window.clearTimeout(fallback);
        root.classList.remove("rp-startup-running");
        document.removeEventListener("keydown", preventBackgroundKeys, true);
        document.removeEventListener("wheel", preventScroll, true);
        document.removeEventListener("touchmove", preventScroll, true);
        window.removeEventListener("pagehide", finish);
        reducedMotion.removeEventListener("change", finish);
        document.removeEventListener("DOMContentLoaded", protectPage);
        for (const [element, wasInert] of covered) element.inert = wasInert;
        covered.clear();
        splash.remove();
    }

    function preventBackgroundKeys(event) {
        // Keep focus and scroll keys from acting on the covered application.
        if (["Tab", " ", "ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", "Enter"].includes(event.key)) {
            event.preventDefault();
        }
        if (event.key === "Escape") finish();
    }

    // Install a fallback before locking scroll, including if animations are disabled.
    fallback = window.setTimeout(finish, reducedMotion.matches ? 500 : 3700);
    splash.addEventListener("animationend", event => {
        if (event.target === splash) finish();

    });
    splash.addEventListener("animationcancel", event => {
        if (event.target === splash) finish();
    });
    window.addEventListener("pagehide", finish);
    reducedMotion.addEventListener("change", finish);
    document.addEventListener("keydown", preventBackgroundKeys, true);
    document.addEventListener("wheel", preventScroll, { capture: true, passive: false });
    document.addEventListener("touchmove", preventScroll, { capture: true, passive: false });
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", protectPage, { once: true });
    } else {
        protectPage();
    }
    root.classList.add("rp-startup-running");
    splash.classList.add("rp-startup--active");

    // Fail open if the stylesheet is unavailable or animations are overridden.
    if (window.getComputedStyle(splash).animationName === "none") finish();
})();
