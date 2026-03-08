// MediClaims AI — Post-Submission Agentic Appeals Dashboard
// AWS Agentic AI Hackathon · Bedrock Nova Micro · 6-Agent Pipeline

class PatientsManager {
    constructor() {
        this.patients = [];
        this.filtered = [];
        this.init();
    }

    async init() {
        this.setupFilters();
        this.showLoading();
        await Promise.all([this.loadPatients(), this.loadPipeline()]);
        this.hideLoading();
    }

    setupFilters() {
        ['priorityFilter','payerFilter'].forEach(id => {
            document.getElementById(id)?.addEventListener('change', () => this.applyFilters());
        });
        document.getElementById('searchFilter')?.addEventListener('input', () => this.applyFilters());
    }

    showLoading() { const el = document.getElementById('loadingOverlay'); if (el) el.style.display = 'flex'; }
    hideLoading() { const el = document.getElementById('loadingOverlay'); if (el) el.style.display = 'none'; }

    async loadPatients() {
        try {
            const r = await fetch('/appeals/api/denied-claims');
            if (!r.ok) throw new Error('HTTP ' + r.status);
            const d = await r.json();
            this.patients = Array.isArray(d.denied_claims) ? d.denied_claims : [];
        } catch (e) {
            console.error('loadPatients:', e);
            this.patients = [];
        }
        this.filtered = [...this.patients];
        this.renderGrid();
        this.renderKPIs();
        this.updateCount();
    }

    async loadPipeline() {
        try {
            const r = await fetch('/appeals/api/pipeline-status');
            if (!r.ok) return;
            const d = await r.json();
            this.renderPipelineStats(d.pipeline_summary || {});
            this.renderActivityFeed(d.recent_agent_activity || []);
            this.renderStatusList(d.claim_statuses || {});
            // Show Bedrock badge in header
            if ((d.pipeline_summary || {}).bedrock_available) {
                const b = document.getElementById('bedrockBadge');
                if (b) b.style.display = 'inline-flex';
                const p = document.getElementById('bedrockPill');
                if (p) { p.textContent = 'Bedrock Active'; p.style.background = 'rgba(255,255,255,.3)'; }
            } else {
                const p = document.getElementById('bedrockPill');
                if (p) p.textContent = 'Bedrock Offline';
            }
        } catch (e) {
            console.warn('loadPipeline:', e);
            const p = document.getElementById('bedrockPill');
            if (p) p.textContent = 'Unavailable';
        }
    }

    renderPipelineStats(s) {
        const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
        set('ps-approved', s.approved || 0);
        set('ps-denied',   s.denied   || 0);
        set('ps-resub',    s.resubmitted || 0);
        set('ps-appeal',   s.appeal_generated || 0);
    }

    renderActivityFeed(activities) {
        const feed = document.getElementById('activityFeed');
        if (!feed) return;
        if (!activities.length) {
            feed.innerHTML = '<div class="activity-loading"><span>No recent activity</span></div>';
            return;
        }
        const icons = {
            'Risk Predictor': '&#129504;', 'Auto Corrector': '&#128295;',
            'Claim Submitter': '&#128228;', 'Appeal Generator': '&#128221;',
            'Resubmitter': '&#128260;', 'Feedback Learner': '&#128200;', 'Pipeline': '&#9881;'
        };
        feed.innerHTML = activities.slice(0, 10).map(a => {
            const icon = icons[a.agent] || '&#9881;';
            const ts = a.timestamp ? new Date(a.timestamp).toLocaleString('en-US', {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}) : '';
            const risk = a.risk_score != null ? a.risk_score : null;
            const riskColor = risk > .6 ? 'var(--danger)' : risk > .3 ? 'var(--warning)' : 'var(--success)';
            const detail = (a.claim_id || a.patient_id || '') + (a.final_status ? ' · ' + a.final_status.replace(/_/g,' ') : '');
            return '<div class="activity-item">'
                + '<span class="activity-icon">' + icon + '</span>'
                + '<div class="activity-body">'
                + '<div class="activity-agent">' + a.agent + '</div>'
                + (detail ? '<div class="activity-detail">' + detail + '</div>' : '')
                + (risk != null ? '<div class="activity-risk" style="color:' + riskColor + '">Risk ' + Math.round(risk*100) + '%</div>' : '')
                + '</div>'
                + '<div class="activity-time">' + ts + '</div>'
                + '</div>';
        }).join('');
    }

    renderStatusList(statuses) {
        const list = document.getElementById('claimStatusList');
        if (!list) return;
        const entries = Object.entries(statuses).slice(0, 8);
        if (!entries.length) { list.innerHTML = '<div class="activity-loading"><span>No data</span></div>'; return; }
        const cls = (s) => {
            if (!s) return 'other';
            if (s === 'approved') return 'approved';
            if (s.includes('resubmit')) return 'resubmitted';
            if (s.includes('appeal')) return 'appeal';
            if (s === 'denied' || s === 'rejected') return 'denied';
            return 'other';
        };
        list.innerHTML = entries.map(([pid, s]) => {
            const name = (s.patient_name || pid);
            const status = (s.status || 'unknown').replace(/_/g,' ');
            return '<div class="status-row">'
                + '<span class="status-name">' + name + '</span>'
                + '<span class="status-badge ' + cls(s.status) + '">' + status + '</span>'
                + '</div>';
        }).join('');
    }

    renderGrid() {
        const grid = document.getElementById('patientsGrid');
        const empty = document.getElementById('emptyState');
        if (!grid || !empty) return;
        if (!this.filtered.length) {
            grid.style.display = 'none';
            empty.style.display = 'block';
            return;
        }
        grid.style.display = 'grid';
        empty.style.display = 'none';
        grid.innerHTML = this.filtered.map(p => this.buildCard(p)).join('');
    }

    buildCard(p) {
        const initials = (p.name || 'NA').split(' ').map(n => n[0]).join('').slice(0,2).toUpperCase();
        const amt = new Intl.NumberFormat('en-US', {style:'currency',currency:'USD'}).format(p.amount || 0);
        const risk = Math.round((p.riskScore || 0) * 100);
        const riskClass = risk > 60 ? 'risk-high' : risk > 30 ? 'risk-med' : 'risk-low';
        const svcDate = p.serviceDate ? new Date(p.serviceDate).toLocaleDateString('en-US',{month:'short',day:'numeric',year:'numeric'}) : 'N/A';
        const insurer = (p.payer || '').toLowerCase();
        const logoMap = {aetna:'AET',united:'UHC',bluecross:'BCBS',cigna:'CIG'};
        const logo = logoMap[insurer] || 'INS';

        return '<div class="claim-card" onclick="viewDetails(\'' + p.id + '\')">'
            + '<div class="card-top">'
            + '<div class="card-patient">'
            + '<div class="avatar">' + initials + '</div>'
            + '<div><div class="patient-name">' + (p.name || 'Unknown') + '</div>'
            + '<div class="patient-sub">' + p.id + ' · ' + (p.claimId || '') + '</div></div>'
            + '</div>'
            + '<span class="priority-pill ' + (p.priority || 'low') + '">' + (p.priority || 'low') + '</span>'
            + '</div>'
            + '<div class="card-fields">'
            + '<div><div class="cf-label">Amount</div><div class="cf-value amount">' + amt + '</div></div>'
            + '<div><div class="cf-label">Service Date</div><div class="cf-value">' + svcDate + '</div></div>'
            + '<div><div class="cf-label">AI Risk Score</div><div class="cf-value ' + riskClass + '">' + risk + '%</div></div>'
            + '<div><div class="cf-label">Appeal Success</div><div class="cf-value">' + (p.successProbability || 0) + '%</div></div>'
            + '</div>'
            + '<div class="denial-strip">'
            + '<div class="denial-code-row"><span class="denial-code-badge">' + (p.denialCode || 'CO-16') + '</span><span class="denial-code-label">' + (p.denialCategory || '').replace(/_/g,' ') + '</span></div>'
            + '<div class="denial-reason-text">' + (p.denialReason || 'Claim denied') + '</div>'
            + '</div>'
            + '<div class="card-insurer">'
            + '<div class="insurer-logo">' + logo + '</div>'
            + '<div><div class="insurer-name">' + (p.payerName || 'Unknown') + '</div>'
            + '<div class="insurer-proc">' + (p.procedure || '') + '</div></div>'
            + '</div>'
            + '<div class="card-actions">'
            + '<button class="btn-details" onclick="event.stopPropagation();viewDetails(\'' + p.id + '\')">'
            + '<i class="fas fa-search"></i> Details &amp; ERA'
            + '</button>'
            + '<button class="btn-ai-appeal" onclick="event.stopPropagation();genAppeal(\'' + p.id + '\',\'' + (p.name||'').replace(/'/g,'') + '\')">'
            + '&#129504; AI Appeal'
            + '</button>'
            + '</div>'
            + '</div>';
    }

    renderKPIs() {
        const urgent = this.patients.filter(p => p.priority === 'high').length;
        const total  = this.patients.length;
        const amount = this.patients.reduce((s, p) => s + (p.amount || 0), 0);
        const avg    = total ? Math.round(this.patients.reduce((s,p) => s + (p.successProbability||0), 0) / total) : 0;
        const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
        set('urgentClaims',  urgent);
        set('totalDenied',   total);
        set('recoveryAmount','$' + Math.round(amount / 1000) + 'K');
        set('avgSuccess',    avg + '%');
    }

    updateCount() {
        const el = document.getElementById('claimCount');
        if (el) el.textContent = this.filtered.length + ' claim' + (this.filtered.length !== 1 ? 's' : '');
    }

    applyFilters() {
        const pri  = document.getElementById('priorityFilter')?.value || '';
        const pay  = document.getElementById('payerFilter')?.value || '';
        const srch = (document.getElementById('searchFilter')?.value || '').toLowerCase();
        this.filtered = this.patients.filter(p => {
            if (pri && p.priority !== pri) return false;
            if (pay && !(p.payer||'').toLowerCase().includes(pay)) return false;
            if (srch && ![(p.name||''),(p.claimId||''),(p.id||''),(p.denialReason||'')].some(v => v.toLowerCase().includes(srch))) return false;
            return true;
        });
        this.renderGrid();
        this.updateCount();
    }

    toast(msg, type) {
        type = type || 'info';
        const c = document.getElementById('toastContainer');
        if (!c) return;
        const t = document.createElement('div');
        t.className = 'toast toast-' + type;
        t.innerHTML = '<i class="fas fa-' + (type==='success'?'check-circle':'info-circle') + '"></i><span>' + msg + '</span>';
        c.appendChild(t);
        setTimeout(() => { if (t.parentNode) t.parentNode.removeChild(t); }, 4000);
    }
}

// ── Globals ──────────────────────────────────────────────────────────────────

function viewDetails(id) { window.location.href = '/appeals/patient-details/' + id; }

function loadPatients() {
    if (window.pm) { window.pm.toast('Refreshing…', 'info'); window.pm.loadPatients(); }
}

function clearFilters() {
    ['priorityFilter','payerFilter'].forEach(id => { const el = document.getElementById(id); if (el) el.value = ''; });
    const s = document.getElementById('searchFilter'); if (s) s.value = '';
    if (window.pm) window.pm.applyFilters();
}

async function genAppeal(patientId, patientName) {
    const btn = event.currentTarget;
    const orig = btn.innerHTML;
    btn.innerHTML = '&#9203; Generating…';
    btn.disabled = true;
    try {
        const r = await fetch('/appeals/api/bedrock-appeal/' + patientId, {method:'POST'});
        if (!r.ok) throw new Error('HTTP ' + r.status);
        const d = await r.json();
        showAppealModal(patientName, d);
        if (window.pm) window.pm.toast('Appeal generated for ' + patientName, 'success');
    } catch (e) {
        if (window.pm) window.pm.toast('Appeal generation failed', 'error');
    } finally {
        btn.innerHTML = orig;
        btn.disabled = false;
    }
}

function showAppealModal(name, data) {
    const ex = document.getElementById('appealModal');
    if (ex) ex.remove();
    const m = document.createElement('div');
    m.id = 'appealModal';
    m.style.cssText = 'position:fixed;inset:0;background:rgba(15,23,42,.6);z-index:300;display:flex;align-items:center;justify-content:center;padding:16px';
    const letter = (data.appeal_letter || 'No letter generated').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    m.innerHTML = '<div style="background:#fff;border-radius:14px;max-width:660px;width:100%;max-height:88vh;overflow-y:auto;box-shadow:0 25px 50px rgba(0,0,0,.25)">'
        + '<div style="padding:16px 20px;border-bottom:1px solid #e2e8f0;display:flex;justify-content:space-between;align-items:flex-start">'
        + '<div><div style="font-size:14px;font-weight:700;color:#1e293b">AI-Generated Appeal Letter</div>'
        + '<div style="font-size:11px;color:#94a3b8;margin-top:2px">Powered by ' + (data.generated_by || 'AWS Bedrock Nova Micro') + '</div></div>'
        + '<button onclick="document.getElementById(\'appealModal\').remove()" style="background:none;border:none;font-size:20px;cursor:pointer;color:#94a3b8;line-height:1;padding:0 4px">&times;</button>'
        + '</div>'
        + '<div style="padding:20px">'
        + '<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px">'
        + '<span style="background:#fee2e2;color:#991b1b;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700">' + (data.denial_code||'') + '</span>'
        + '<span style="background:#dbeafe;color:#1e40af;padding:3px 10px;border-radius:20px;font-size:11px">' + (data.claim_id||'') + '</span>'
        + '<span style="background:#dcfce7;color:#166534;padding:3px 10px;border-radius:20px;font-size:11px">' + name + '</span>'
        + '</div>'
        + '<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:16px;font-size:13px;line-height:1.7;color:#374151;white-space:pre-wrap;font-family:Georgia,serif">' + letter + '</div>'
        + '<div style="margin-top:14px;display:flex;gap:8px;justify-content:flex-end">'
        + '<button onclick="copyAppeal()" style="padding:7px 14px;border:1px solid #e2e8f0;border-radius:6px;background:#fff;cursor:pointer;font-size:12px;font-weight:500">&#128203; Copy</button>'
        + '<button onclick="document.getElementById(\'appealModal\').remove()" style="padding:7px 14px;border:none;border-radius:6px;background:#2563eb;color:#fff;cursor:pointer;font-size:12px;font-weight:600">Close</button>'
        + '</div></div></div>';
    document.body.appendChild(m);
    m.addEventListener('click', e => { if (e.target === m) m.remove(); });
}

function copyAppeal() {
    const m = document.getElementById('appealModal');
    if (!m) return;
    const el = m.querySelector('[style*="pre-wrap"]');
    if (el) navigator.clipboard.writeText(el.textContent).then(() => { if (window.pm) window.pm.toast('Copied!', 'success'); });
}

document.addEventListener('DOMContentLoaded', () => { window.pm = new PatientsManager(); });
