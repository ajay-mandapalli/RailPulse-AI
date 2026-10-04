(() => {
 "use strict";
 const menus = [...document.querySelectorAll(".rp-account")];
 document.addEventListener("click", event => {
  menus.forEach(menu => { if (!menu.contains(event.target)) menu.open = false; });
 });
 document.addEventListener("keydown", event => {
  if (event.key === "Escape") menus.forEach(menu => {
   if (menu.open) { menu.open = false; menu.querySelector("summary").focus(); }
  });
 });
 const form = document.getElementById("predictionForm");
 const overlay = document.getElementById("rp-processing");
 const button = document.getElementById("generateButton");
 if (!form || !overlay || !button) return;
 const original = button.innerHTML;
 let previousOverflow;
 let previousInert;
 const content = document.querySelector(".app-layout");
 function restore() {
  overlay.hidden = true;
  button.disabled = false;
  button.innerHTML = original;
  form.removeAttribute("aria-busy");
  if (previousOverflow !== undefined) document.body.style.overflow = previousOverflow;
  if (content && previousInert !== undefined) content.inert = previousInert;
 }
 form.addEventListener("submit", event => {
  if (event.defaultPrevented || !form.checkValidity()) return;
  previousOverflow = document.body.style.overflow;
  previousInert = content?.inert;
  overlay.hidden = false;
  button.disabled = true;
  button.textContent = "Generating forecast...";
  form.setAttribute("aria-busy", "true");
  document.body.style.overflow = "hidden";
  if (content) content.inert = true;
 });
 window.addEventListener("pageshow", restore);
 window.addEventListener("pagehide", restore);
})();
