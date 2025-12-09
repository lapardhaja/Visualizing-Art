// --- utility: parse JSON that may contain bare NaN/Infinity ---
async function fetchJsonClean(url) {
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
    const raw = await res.text();
    const cleaned = raw
        .replace(/\bNaN\b/g, 'null')
        .replace(/\bInfinity\b/g, 'null')
        .replace(/\b-Infinity\b/g, 'null');
    return JSON.parse(cleaned);
}

// --- value pretty-printer ---
function fmt(value) {
    if (value === null || value === undefined || value === '' || value === ' ') return 'N/A';
    if (Array.isArray(value)) {
        if (value.length === 0) return 'N/A';
        // try to show meaningful fields for arrays of objects
        const mapped = value.map(v => {
            if (v && typeof v === 'object') {
                return v.term || v.name || v.role || v.elementName || JSON.stringify(v);
            }
            return String(v);
        });
        return mapped.join(', ');
    }
    if (typeof value === 'object') {
        // show compact JSON for objects
        return JSON.stringify(value);
    }
    return String(value);
}

// --- render a key/value table from an object ---
function renderTable(obj, tbody) {
    tbody.innerHTML = '';
    const entries = Object.entries(obj || {});
    if (entries.length === 0) {
        tbody.innerHTML = '<tr><td colspan="2">N/A</td></tr>';
        return;
    }
    for (const [k, v] of entries) {
        const tr = document.createElement('tr');
        const tdK = document.createElement('td');
        const tdV = document.createElement('td');
        tdK.textContent = k;
        tdV.textContent = fmt(v);
        tr.append(tdK, tdV);
        tbody.appendChild(tr);
    }
}

// helper: read ?id=#### from URL, default to 466
function getObjectIdFromQuery() {
    const params = new URLSearchParams(window.location.search);
    const id = params.get('id');
    if (!id) return '466';
    return id;
}

// --- main loader ---
(async () => {
    const objectId = getObjectIdFromQuery();

    try {
        const data = await fetchJsonClean(`../../metadata/${objectId}.json`);
        const excel = data.excel_metadata || {};
        const api = data.api_metadata || {};
        const id = data.object_id || objectId;

        // image
        const img = document.getElementById('art-img');
        img.src = `../../images/${id}.jpg`;
        img.loading = 'eager';
        img.decoding = 'async';

        // caption
        const title = excel.Title || 'Untitled';
        const artist = excel['Artist Display Name'] || 'N/A';
        const date = excel['Object Date'] || 'N/A';
        document.getElementById('art-caption').innerHTML =
            `<strong>${title}</strong><br>${artist} — ${date}`;

        // tables
        renderTable(excel, document.querySelector('#excel-table tbody'));
        renderTable(api, document.querySelector('#api-table tbody'));
    } catch (e) {
        console.error(e);
        const cap = document.getElementById('art-caption');
        cap.textContent = 'Failed to load artwork metadata.';
    }
})();