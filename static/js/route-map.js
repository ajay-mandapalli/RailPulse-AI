"use strict";
(() => {
    const routes = JSON.parse(document.getElementById("route-map-data").textContent);
    const stations = JSON.parse(document.getElementById("station-location-data").textContent);
    const geometry = JSON.parse(document.getElementById("route-geometry-data").textContent);
    const colors = {Low: "#486b3c", Medium: "#996a12", High: "#b34432"};
    const stationFilter = document.getElementById("station-filter");
    const demandFilter = document.getElementById("demand-filter");
    const detail = document.getElementById("route-detail");
    const status = document.getElementById("map-load-status");
    const rows = Array.from(document.querySelectorAll(".map-route-row"));
    let map, markerLayer, routeLayer, selectedLayer, selected = null;
    const routeLines = new Map();
    let visibleRoutes = routes;
    const location = name => [stations[name].lat, stations[name].lon];
    function message(text) { status.textContent = text; status.hidden = !text; }
    function textNode(tag, text) {
        const node = document.createElement(tag); node.textContent = text; return node;
    }
    function fitMap() {
        if (!map) return;
        const names = [...new Set(visibleRoutes.flatMap(route => [route.origin, route.destination]))];
        const points = visibleRoutes.flatMap(route => routePoints(route));
        if (!points.length) points.push(...names.filter(name => stations[name]).map(location));
        if (points.length) map.fitBounds(points, {padding: [35, 35], maxZoom: 14});
    }
    function selectRoute(index) {
        selected = index;
        const route = routes[index];
        detail.textContent = `${route.origin} to ${route.destination}: ${route.demand} demand, ${route.passengers} average passengers per observation, based on ${route.observations.toLocaleString()} observations.`;
        rows.forEach((row, i) => { row.classList.toggle("selected", i === index); row.setAttribute("aria-pressed", String(i === index)); });
        if (!map) return;
        selectedLayer.clearLayers();
        routeLines.forEach((line, routeIndex) => line.setStyle({weight: routeIndex === index ? 7 : 4, opacity: routeIndex === index ? 1 : 0.65}));
        const selectedLine = routeLines.get(index);
        if (selectedLine) selectedLine.bringToFront();
        [route.origin, route.destination].forEach((name, i) => {
            if (!stations[name]) return;
            const icon = L.divIcon({className: "route-endpoint-icon", html: i === 0 ? "A" : "B", iconSize: [32, 32], iconAnchor: [16, 16]});
            const popup = textNode("div", `${i === 0 ? "Departure" : "Arrival"}: ${name}. ${route.passengers} average passengers on this route (${route.demand} demand).`);
            L.marker(location(name), {icon, title: `${i === 0 ? "Departure" : "Arrival"}: ${name}`, zIndexOffset: 1000}).bindPopup(popup).addTo(selectedLayer);
        });
        const points = routePoints(route);
        if (points.length) map.fitBounds(points, {padding: [70, 70], maxZoom: 14});
        else detail.textContent += " Railway geometry is unavailable for this route.";
    }
    function routePoints(route) {
        const path = geometry[`${route.origin}|${route.destination}`];
        return path && path.coordinates.length > 1 ? path.coordinates.map(point => L.latLng(point)) : [];
    }
    function routeArrow(points) {
        const lengths = points.slice(1).map((point, i) => points[i].distanceTo(point));
        const halfway = lengths.reduce((sum, length) => sum + length, 0) / 2;
        let traversed = 0;
        for (let i = 0; i < lengths.length; i++) {
            if (lengths[i] > 0 && traversed + lengths[i] >= halfway) {
                const fraction = (halfway - traversed) / lengths[i];
                const before = map.latLngToLayerPoint(points[i]);
                const after = map.latLngToLayerPoint(points[i + 1]);
                const center = L.point(before.x + fraction * (after.x - before.x), before.y + fraction * (after.y - before.y));
                const angle = Math.atan2(after.y - before.y, after.x - before.x);
                const wings = [-0.55, 0.55].map(offset => map.layerPointToLatLng(L.point(center.x - 8 * Math.cos(angle + offset), center.y - 8 * Math.sin(angle + offset))));
                return [wings[0], map.layerPointToLatLng(center), wings[1]];
            }
            traversed += lengths[i];
        }
        return [];
    }
    function drawRoutes() {
        if (!map) return;
        routeLayer.clearLayers(); routeLines.clear();
        routes.forEach((route, index) => {
            if (rows[index].hidden || !stations[route.origin] || !stations[route.destination]) return;
            const points = routePoints(route);
            if (points.length < 2) return;
            const description = `${route.origin} to ${route.destination}: ${route.demand} demand, ${route.passengers} average passengers`;
            const popup = document.createElement("div"); popup.className = "station-popup";
            popup.append(textNode("strong", `${route.origin} to ${route.destination}`), textNode("p", `${route.demand} demand - ${route.passengers} average passengers per observation. Based on ${route.observations.toLocaleString()} observations.`));
            const line = L.polyline(points, {color: colors[route.demand], weight: selected === index ? 7 : 4, opacity: selected === index ? 1 : 0.75, className: "demand-route-line"})
                .bindTooltip(textNode("span", description), {sticky: true})
                .bindPopup(popup).addTo(routeLayer);
            line.on("click", () => selectRoute(index));
            routeLines.set(index, line);
            const arrow = routeArrow(points);
            if (arrow.length) L.polyline(arrow, {color: colors[route.demand], weight: 2, opacity: 1, interactive: false, className: "route-direction-arrow"}).addTo(routeLayer);
        });
        const activeLine = routeLines.get(selected);
        if (activeLine) activeLine.bringToFront();
    }
    function drawStations() {
        if (!map) return;
        markerLayer.clearLayers();
        const names = [...new Set(visibleRoutes.flatMap(route => [route.origin, route.destination]))];
        names.forEach(name => {
            if (!stations[name]) return;
            const popup = document.createElement("div"); popup.className = "station-popup";
            popup.appendChild(textNode("strong", name));
            const button = textNode("button", "Show routes at this station"); button.type = "button";
            button.addEventListener("click", () => { stationFilter.value = name; filterRoutes(); map.closePopup(); });
            popup.appendChild(button);
            L.circleMarker(location(name), {radius: 4, color: "#304e3b", weight: 2, fillColor: "#fffdf8", fillOpacity: 1, className: "route-station-point"})
                .bindTooltip(textNode("span", name), {direction: "top"}).bindPopup(popup).addTo(markerLayer);
        });
    }
    function filterRoutes() {
        visibleRoutes = [];
        routes.forEach((route, index) => {
            const visible = (!stationFilter.value || route.origin === stationFilter.value || route.destination === stationFilter.value) && (!demandFilter.value || route.demand === demandFilter.value);
            rows[index].hidden = !visible;
            if (visible) visibleRoutes.push(route);
        });
        if (selected !== null && rows[selected].hidden) {
            rows[selected].classList.remove("selected"); rows[selected].setAttribute("aria-pressed", "false");
            selected = null;
            if (selectedLayer) selectedLayer.clearLayers();
            detail.textContent = "Select a route to inspect its average passenger demand.";
        }
        document.getElementById("route-count").textContent = visibleRoutes.length;
        document.getElementById("highest-demand").textContent = Math.max(0, ...visibleRoutes.map(route => route.passengers));
        document.getElementById("route-list-status").textContent = `${visibleRoutes.length} directional routes`;
        document.getElementById("map-empty").hidden = visibleRoutes.length > 0;
        drawStations();
        fitMap();
        drawRoutes();
    }
    rows.forEach((row, index) => row.addEventListener("click", () => selectRoute(index)));
    stationFilter.addEventListener("change", filterRoutes);
    demandFilter.addEventListener("change", filterRoutes);
    if (typeof L === "undefined") {
        message("The city map could not start. Reload the page to try again. You can still explore demand using the filters and route list below.");
        document.querySelectorAll(".map-controls button").forEach(button => { button.disabled = true; });
        return;
    }
    map = L.map("city-map", {zoomControl: false, scrollWheelZoom: false, minZoom: 9, maxZoom: 19}).setView([17.44, 78.46], 11);
    L.control.scale({imperial: false}).addTo(map);
    const tiles = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    });
    let tileErrors = false;
    tiles.on("loading", () => { tileErrors = false; });
    tiles.on("tileerror", () => { tileErrors = true; message("Some street-map tiles could not load. Check your internet connection, then move or zoom the map to retry. Demand filters and the route list remain available."); });
    tiles.on("load", () => { if (!tileErrors) message(""); });
    tiles.addTo(map);
    routeLayer = L.layerGroup().addTo(map);
    markerLayer = L.layerGroup().addTo(map);
    selectedLayer = L.layerGroup().addTo(map);
    document.getElementById("map-zoom-in").addEventListener("click", () => map.zoomIn());
    document.getElementById("map-zoom-out").addEventListener("click", () => map.zoomOut());
    document.getElementById("map-fit").addEventListener("click", fitMap);
    map.on("zoomend", () => {
        drawRoutes();
        document.getElementById("map-zoom-in").disabled = map.getZoom() >= map.getMaxZoom();
        document.getElementById("map-zoom-out").disabled = map.getZoom() <= map.getMinZoom();
    });
    filterRoutes();
})();
