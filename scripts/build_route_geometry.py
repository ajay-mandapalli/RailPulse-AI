"""Build station-pair paths along OSM railway edges. Run from the project root.

Input: data/mmts_railway_osm.json (Overpass railway=rail ways, out geom).
Only connected rail edges are used; no straight-line gap filling is allowed.
Paths are shortest mapped rail paths, not a timetable or platform assignment.
"""
import csv
import heapq
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def distance(a, b):
    return math.hypot((a[0] - b[0]) * 111195, (a[1] - b[1]) * 106000)

def build():
    raw = json.loads((ROOT / 'data/mmts_railway_osm.json').read_text(encoding='utf-8'))
    stations = json.loads((ROOT / 'static/data/station_locations.json').read_text(encoding='utf-8'))
    graph = defaultdict(dict)
    coordinates = {}
    edge_ways = {}
    for way in raw['elements']:
        tags = way.get('tags', {})
        if tags.get('service') in ('siding', 'yard', 'spur') or tags.get('usage') in ('military', 'siding'):
            continue
        nodes = way['nodes']
        points = [(p['lat'], p['lon']) for p in way['geometry']]
        coordinates.update(zip(nodes, points))
        for a, b, pa, pb in zip(nodes, nodes[1:], points, points[1:]):
            graph[a][b] = graph[b][a] = distance(pa, pb)
            edge_ways[a, b] = edge_ways[b, a] = way['id']
    # Ignore isolated fragments: every station must snap to the connected network.
    remaining, components = set(graph), []
    while remaining:
        pending = [remaining.pop()]
        component = set(pending)
        while pending:
            for neighbor in graph[pending.pop()]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    pending.append(neighbor)
        components.append(component)
    connected_nodes = max(components, key=len)
    station_nodes = {}
    for name, station in stations.items():
        point = (station['lat'], station['lon'])
        node = min(connected_nodes, key=lambda n: distance(coordinates[n], point))
        gap = distance(coordinates[node], point)
        if gap > 200:
            raise ValueError(f'{name}: nearest track is {gap:.0f} m away')
        station_nodes[name] = node
    with (ROOT / 'data/processed_train.csv').open(encoding='utf-8', newline='') as file:
        pairs = sorted({(r['from_station'], r['to_station']) for r in csv.DictReader(file)})
    paths = {}
    for origin, destination in pairs:
        start, end = station_nodes[origin], station_nodes[destination]
        distances, previous, queue = {start: 0}, {}, [(0, start)]
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != distances[node]:
                continue
            if node == end:
                break
            for neighbor, weight in graph[node].items():
                candidate = cost + weight
                if candidate < distances.get(neighbor, float('inf')):
                    distances[neighbor], previous[neighbor] = candidate, node
                    heapq.heappush(queue, (candidate, neighbor))
        if end not in distances:
            raise ValueError(f'No connected rail path: {origin} -> {destination}')
        nodes = [end]
        while nodes[-1] != start:
            nodes.append(previous[nodes[-1]])
        nodes.reverse()
        assert len(nodes) >= 2
        paths[origin + '|' + destination] = {
            'coordinates': [coordinates[node] for node in nodes],
            'distance_km': round(distances[end] / 1000, 2),
            'osm_way_ids': sorted({edge_ways[a, b] for a, b in zip(nodes, nodes[1:])}),
        }
    result = {
        'source': 'OpenStreetMap contributors, ODbL 1.0',
        'source_url': 'https://www.openstreetmap.org/copyright',
        'osm_timestamp': raw.get('osm3s', {}).get('timestamp_osm_base'),
        'method': 'Shortest connected rail path; excludes sidings, yards, spurs and military tracks. Station snaps limited to 200 metres. No synthetic connecting edges.',
        'routes': paths,
    }
    target = ROOT / 'static/data/route_geometry.json'
    target.write_text(json.dumps(result, separators=(',', ':')) + '\n', encoding='utf-8')
    print(f'Built {len(paths)} railway paths in {target.name}')
    for key in ['Lingampalli|Falaknuma', 'Secunderabad Jn|Medchal', 'Bolarum Bazar|Medchal', 'Bharat Nagar|RC Puram']:
        if key in paths:
            print(key, paths[key]['distance_km'], 'km', len(paths[key]['coordinates']), 'track points')

if __name__ == '__main__':
    build()
