/* Show on the first visit and explicit sign-in, signup, or logout transitions. */
(() => {
    "use strict";
    const splash = document.getElementById("rp-startup");
    if (!splash) return;

    // Replay once per tab, then only after an explicit authentication action.
    const seenKey = "railpulse:startup:seen";
    const pendingKey = "railpulse:startup:auth";
    const authPaths = new Set(
        [splash.dataset.loginUrl, splash.dataset.signupUrl, splash.dataset.logoutUrl]
            .map(path => new URL(path, window.location.href).pathname)
    );

    function isAuthUrl(value) {
        const url = new URL(value, window.location.href);
        return url.origin === window.location.origin && authPaths.has(url.pathname);
    }

    function rememberAuthAction() {
        try {
            window.sessionStorage.setItem(pendingKey, "1");
        } catch (_) {
            // The referrer fallback below supports browsers with storage disabled.
        }
    }

    // These listeners remain after the overlay is removed to identify the next load.
    document.addEventListener("click", event => {
        const link = event.target.closest("a[href]");
        if (!link || event.defaultPrevented || event.button !== 0 ||
            event.ctrlKey || event.metaKey || event.shiftKey || event.altKey ||
            link.hasAttribute("download") || (link.target && link.target !== "_self")) return;
        if (isAuthUrl(link.href)) rememberAuthAction();
    });
    document.addEventListener("submit", event => {
        if (!event.defaultPrevented && isAuthUrl(event.target.action)) rememberAuthAction();
    });

    let shouldShow;
    try {
        shouldShow = !window.sessionStorage.getItem(seenKey) ||
            window.sessionStorage.getItem(pendingKey) === "1";
        window.sessionStorage.setItem(seenKey, "1");
        window.sessionStorage.removeItem(pendingKey);
    } catch (_) {
        const navigation = performance.getEntriesByType("navigation")[0];
        const referrer = document.referrer;
        shouldShow = navigation?.type !== "reload" &&
            (!referrer || new URL(referrer).origin !== window.location.origin ||
             isAuthUrl(referrer) || isAuthUrl(window.location.href));
    }
    if (!shouldShow) {
        splash.remove();
        return;
    }

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
    fallback = window.setTimeout(finish, reducedMotion.matches ? 400 : 3200);
    splash.addEventListener("animationend", event => {
        if (event.target === splash) finish();
        if (event.target.classList.contains("rp-startup__train")) {
            // Restore the page dimensions while the overlay is still opaque.
            // Input remains blocked until the fade finishes and the overlay is removed.
            root.classList.remove("rp-startup-running");
        }
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
    if (!reducedMotion.matches) root.classList.add("rp-startup-running");
    splash.classList.add("rp-startup--active");

    // Fail open if the stylesheet is unavailable or animations are overridden.
    if (window.getComputedStyle(splash).animationName === "none") finish();
})();
