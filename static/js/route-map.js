"use strict";
(() => {
    const routes = JSON.parse(document.getElementById("route-map-data").textContent);
    // Deliberate schematic positions, not geographic coordinates.
    const stations = {
        "RC Puram": [120, 150], "Tellapur": [120, 270], "Lingampalli": [250, 210],
        "Hafeezpet": [270, 340], "Hitec City": [190, 430], "Borabanda": [370, 390],
        "Bharat Nagar": [390, 270], "Fateh Nagar": [380, 510], "Begumpet": [490, 450],
        "Necklace Road": [460, 600], "James Street": [570, 550],
        "Secunderabad Jn": [670, 460], "Malkajgiri": [770, 340], "Alwal": [690, 240],
        "Bolarum": [610, 150], "Bolarum Bazar": [810, 180], "Medchal": [810, 65],
        "Charlapalli": [990, 340], "Sitafalmandi": [830, 490], "Arts College": [960, 550],
        "Jamia Osmania": [870, 620], "Vidyanagar": [760, 660], "Kacheguda": [650, 710],
        "Hyderabad Deccan": [410, 760], "Malakpet": [750, 790], "Dabirpura": [890, 790],
        "Yakutpura": [910, 895], "Huppuguda": [740, 930], "Falaknuma": [540, 905]
    };
    const colors = {Low: "#486b3c", Medium: "#996a12", High: "#b34432"};
    const svgNS = "http://www.w3.org/2000/svg";
    const svg = document.getElementById("network-map");
    const stationFilter = document.getElementById("station-filter");
    const demandFilter = document.getElementById("demand-filter");
    const detail = document.getElementById("route-detail");
    const rows = Array.from(document.querySelectorAll(".map-route-row"));
    const paths = [];
    let selected = null;
    function element(tag, attributes) {
        const node = document.createElementNS(svgNS, tag);
        Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
        return node;
    }
    function describe(route) {
        return `${route.origin} to ${route.destination}: ${route.demand} demand, ${route.passengers} average passengers`;
    }
    function selectRoute(index) {
        selected = index;
        const route = routes[index];
        detail.textContent = `${describe(route)} per observation, based on ${route.observations.toLocaleString()} observations. Direction: ${route.origin} → ${route.destination}.`;
        rows.forEach((row, i) => { row.classList.toggle("selected", i === index); row.setAttribute("aria-pressed", String(i === index)); });
        paths.forEach((path, i) => { if (path) path.classList.toggle("selected", i === index); });
        if (paths[index]) paths[index].parentNode.appendChild(paths[index]);
    }
    routes.forEach((route, index) => {
        const a = stations[route.origin], b = stations[route.destination];
        if (!a || !b) { paths.push(null); return; }
        const dx = b[0] - a[0], dy = b[1] - a[1];
        const length = Math.hypot(dx, dy) || 1;
        // Reversed directions curve on opposite sides, so both remain selectable.
        const bend = Math.min(90, length * 0.18);
        const cx = (a[0] + b[0]) / 2 - dy / length * bend;
        const cy = (a[1] + b[1]) / 2 + dx / length * bend;
        const path = element("path", {d: `M ${a[0]} ${a[1]} Q ${cx} ${cy} ${b[0]} ${b[1]}`, stroke: colors[route.demand], class: "network-route", "aria-label": describe(route)});
        const title = element("title", {}); title.textContent = describe(route); path.appendChild(title);
        path.addEventListener("click", () => selectRoute(index));
        document.getElementById("map-routes").appendChild(path); paths.push(path);
    });
    Object.entries(stations).forEach(([name, [x, y]]) => {
        const group = element("g", {class: "network-station", transform: `translate(${x}, ${y})`});
        group.appendChild(element("circle", {r: 7}));
        const label = element("text", {y: -15, "text-anchor": "middle"}); label.textContent = name; group.appendChild(label);
        group.addEventListener("click", () => { stationFilter.value = name; filterRoutes(); });
        document.getElementById("map-stations").appendChild(group);
    });
    function filterRoutes() {
        let count = 0, highest = 0;
        routes.forEach((route, index) => {
            const visible = (!stationFilter.value || route.origin === stationFilter.value || route.destination === stationFilter.value) && (!demandFilter.value || route.demand === demandFilter.value);
            rows[index].hidden = !visible;
            if (paths[index]) paths[index].style.display = visible ? "" : "none";
            if (visible) { count++; highest = Math.max(highest, route.passengers); }
        });
        if (selected !== null && rows[selected].hidden) {
            rows[selected].classList.remove("selected"); rows[selected].setAttribute("aria-pressed", "false");
            if (paths[selected]) paths[selected].classList.remove("selected");
            selected = null; detail.textContent = "Select a route to inspect its average passenger demand.";
        }
        document.getElementById("route-count").textContent = count;
        document.getElementById("highest-demand").textContent = highest;
        document.getElementById("route-list-status").textContent = `${count} directional routes`;
        document.getElementById("map-empty").hidden = count > 0;
    }
    rows.forEach((row, index) => row.addEventListener("click", () => selectRoute(index)));
    stationFilter.addEventListener("change", filterRoutes);
    demandFilter.addEventListener("change", filterRoutes);
    let zoom = 1;
    function setZoom(value) {
        zoom = Math.max(1, Math.min(3, value)); svg.style.width = `${zoom * 100}%`;
        document.getElementById("map-zoom-in").disabled = zoom >= 3;
        document.getElementById("map-zoom-out").disabled = zoom <= 1;
    }
    document.getElementById("map-zoom-in").addEventListener("click", () => setZoom(zoom + 0.5));
    document.getElementById("map-zoom-out").addEventListener("click", () => setZoom(zoom - 0.5));
    document.getElementById("map-fit").addEventListener("click", () => { setZoom(1); document.getElementById("network-viewport").scrollTo(0, 0); });
    setZoom(1);
})();
