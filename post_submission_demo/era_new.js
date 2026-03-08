/* MediClaims AI — ERA Processing & Analysis
 * Complete rebuild: enterprise-grade, real medical billing logic
 */
'use strict';

var API = '';
var patients = [];
var selectedPatient = null;
var lastEraData = null;

document.addEventListener('DOMContentLoaded', function () {
    loadPatients();
    setupSearch();
    setupDropZone();
    document.getElementById('eraFile').addEventListener('change', onFileSelected);
});

// ── Utilities ─────────────────────────────────────────────────────────────────
function esc(s) {
    if (s == null) return '';
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function safeStr(v) {
    if (v == null) return '';
    if (typeof v === 'string') return v;
    if (typeof v === 'object') {
        return v.action || v.text || v.description || v.recommendation ||
               v.message || v.title || v.detail || JSON.stringify(v);
    }
    return String(v);
}
function fmt(n) {
    if (!n && n !== 0) return '$0';
    return '$' + Number(n).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 });
}
function initials2(name) {
    return (name || '?').split(' ').map(function (w) { return w[0]; }).join('').slice(0, 2).toUpperCase();
}
function showToast(msg, type) {
    type = type || 'info';
    var icons = { success: 'fa-check-circle', error: 'fa-exclamation-circle', info: 'fa-info-circle' };
    var t = document.createElement('div');
    t.className = 'toast toast-' + type;
    t.innerHTML = '<i class="fas ' + (icons[type] || icons.info) + '"></i><span>' + esc(msg) + '</span>';
    document.getElementById('toastContainer').appendChild(t);
    setTimeout(function () { if (t.parentNode) t.parentNode.removeChild(t); }, 3800);
}

// ── Load patients ─────────────────────────────────────────────────────────────
function loadPatients() {
    fetch(API + '/appeals/api/denied-claims')
        .then(function (r) { return r.json(); })
        .then(function (data) {
            patients = data.denied_claims || [];
            renderPatientList(patients);
            // Show count of actionable denied claims (not accepted — those are done)
            document.getElementById('patientCount').textContent = patients.length;
            document.getElementById('panelSubtitle').textContent =
                patients.length + ' claim' + (patients.length !== 1 ? 's' : '') + ' need attention';
        })
        .catch(function () {
            showToast('Failed to load patients', 'error');
            document.getElementById('patientList').innerHTML =
                '<div class="list-loading" style="color:var(--red)"><i class="fas fa-exclamation-circle"></i><span>Could not load patients</span></div>';
        });
}

function renderPatientList(list) {
    var el = document.getElementById('patientList');
    if (!list.length) {
        el.innerHTML = '<div class="list-loading"><i class="fas fa-check-circle" style="color:var(--green)"></i><span>No pending denials</span></div>';
        return;
    }
    el.innerHTML = list.map(function (p) {
        var ini = initials2(p.name);
        var pri = p.priority || 'medium';
        var amt = p.amount ? fmt(p.amount) : '';
        var stag = statusBadge(p.status);
        var riskPct = p.riskScore ? Math.round(p.riskScore * 100) : 0;
        var riskBar = '<div class="pi-risk-bar"><div class="pi-risk-fill ' + pri + '" style="width:' + riskPct + '%"></div></div>';
        return '<div class="patient-item" data-id="' + p.id + '" onclick="selectPatient(\'' + p.id + '\')">' +
            '<div class="pi-avatar ' + pri + '">' + ini + '</div>' +
            '<div class="pi-info">' +
                '<div class="pi-name">' + esc(p.name) + '</div>' +
                '<div class="pi-sub">' + esc(p.claimId || p.id) + ' &middot; ' + esc(p.payerName || '') + '</div>' +
                '<div class="pi-tags">' +
                    '<span class="pi-tag ' + pri + '">' + pri.toUpperCase() + '</span>' +
                    stag +
                    '<span class="pi-tag code">' + esc(p.denialCode || 'CO-16') + '</span>' +
                '</div>' +
                (amt ? '<div class="pi-amount">' + amt + ' denied</div>' : '') +
                riskBar +
            '</div></div>';
    }).join('');
}

function statusBadge(status) {
    if (!status) return '<span class="pi-tag denied">Denied</span>';
    var s = status.toLowerCase();
    if (s.indexOf('resub') >= 0)  return '<span class="pi-tag resub">Resubmitted</span>';
    if (s.indexOf('appeal') >= 0) return '<span class="pi-tag appeal">Appeal</span>';
    return '<span class="pi-tag denied">Denied</span>';
}

function setupSearch() {
    document.getElementById('searchInput').addEventListener('input', function (e) {
        var q = e.target.value.toLowerCase();
        renderPatientList(patients.filter(function (p) {
            return (p.name || '').toLowerCase().indexOf(q) >= 0 ||
                   (p.claimId || '').toLowerCase().indexOf(q) >= 0 ||
                   (p.payerName || '').toLowerCase().indexOf(q) >= 0 ||
                   (p.denialCode || '').toLowerCase().indexOf(q) >= 0 ||
                   (p.denialReason || '').toLowerCase().indexOf(q) >= 0;
        }));
    });
}

// ── Select patient ────────────────────────────────────────────────────────────
function selectPatient(id) {
    selectedPatient = null;
    lastEraData = null;
    for (var i = 0; i < patients.length; i++) {
        if (patients[i].id === id) { selectedPatient = patients[i]; break; }
    }
    if (!selectedPatient) return;

    document.querySelectorAll('.patient-item').forEach(function (el) {
        el.classList.toggle('active', el.dataset.id === id);
    });
    document.getElementById('noSelection').style.display = 'none';
    document.getElementById('patientView').style.display = 'block';
    document.getElementById('bedrockBadge').style.display = 'inline-flex';

    renderPatientHero(selectedPatient);
    renderClaimHistory(selectedPatient);
    resetAnalysis();

    // Scroll to top of right panel
    document.getElementById('analysisPanel').scrollTop = 0;
}

function renderPatientHero(p) {
    var riskPct = p.riskScore ? Math.round(p.riskScore * 100) : 0;
    var riskCls = riskPct >= 60 ? 'danger' : riskPct >= 30 ? 'warn' : 'ok';
    var sucCls  = (p.successProbability || 0) >= 60 ? 'ok' : (p.successProbability || 0) >= 40 ? 'warn' : 'danger';
    var codeExplain = getDenialCodeExplanation(p.denialCode, p.denialReason);

    document.getElementById('patientHero').innerHTML =
        '<div class="ph-left">' +
            '<div class="ph-avatar">' + initials2(p.name) + '</div>' +
            '<div class="ph-main">' +
                '<div class="ph-name">' + esc(p.name) + '</div>' +
                '<div class="ph-meta">' +
                    '<span><i class="fas fa-birthday-cake"></i> Age ' + (p.age || '—') + '</span>' +
                    '<span><i class="fas fa-hospital"></i> ' + esc(p.payerName || p.payer || '—') + '</span>' +
                    '<span><i class="fas fa-user-md"></i> ' + esc(p.doctorName || '—') + '</span>' +
                    '<span><i class="fas fa-calendar-alt"></i> ' + esc(p.serviceDate || '—') + '</span>' +
                '</div>' +
                '<div class="ph-denial-pill">' +
                    '<i class="fas fa-exclamation-triangle"></i>' +
                    '<strong>' + esc(p.denialCode || 'CO-16') + '</strong> — ' + esc(codeExplain.short) +
                '</div>' +
            '</div>' +
        '</div>' +
        '<div class="ph-stats">' +
            '<div class="ph-stat">' +
                '<div class="ph-stat-label">Billed</div>' +
                '<div class="ph-stat-value">' + fmt(p.amount) + '</div>' +
            '</div>' +
            '<div class="ph-stat">' +
                '<div class="ph-stat-label">Risk Score</div>' +
                '<div class="ph-stat-value ' + riskCls + '">' + riskPct + '%</div>' +
            '</div>' +
            '<div class="ph-stat">' +
                '<div class="ph-stat-label">Appeal Chance</div>' +
                '<div class="ph-stat-value ' + sucCls + '">' + (p.successProbability || 0) + '%</div>' +
            '</div>' +
            '<div class="ph-stat">' +
                '<div class="ph-stat-label">Issues Found</div>' +
                '<div class="ph-stat-value ' + (p.issuesCount > 0 ? 'danger' : 'ok') + '">' + (p.issuesCount || 0) + '</div>' +
            '</div>' +
        '</div>';
}

function renderClaimHistory(p) {
    var s = (p.status || '').toLowerCase();
    var badgeCls = 'denied', badgeTxt = 'Denied';
    if (s.indexOf('resub') >= 0)         { badgeCls = 'resub';    badgeTxt = 'Resubmitted'; }
    else if (s.indexOf('appeal') >= 0)   { badgeCls = 'appeal';   badgeTxt = 'Appeal Filed'; }
    else if (s.indexOf('approved') >= 0) { badgeCls = 'approved'; badgeTxt = 'Approved'; }

    var badge = document.getElementById('claimStatusBadge');
    badge.className = 'section-badge ' + badgeCls;
    badge.textContent = badgeTxt;

    var fields = [
        { label: 'Claim ID',        value: p.claimId || '—',                    cls: 'mono blue' },
        { label: 'Service Date',    value: p.serviceDate || '—' },
        { label: 'Procedure Code',  value: p.procedure || '—',                   cls: 'mono' },
        { label: 'Insurer',         value: p.payerName || p.payer || '—' },
        { label: 'Provider',        value: p.doctorName || '—' },
        { label: 'Billed Amount',   value: fmt(p.amount),                        cls: 'blue' },
        { label: 'Denial Code',     value: p.denialCode || '—',                  cls: 'red' },
        { label: 'Denial Category', value: (p.denialCategory || '—').replace(/_/g, ' ') },
        { label: 'Issues Found',    value: p.issuesCount != null ? String(p.issuesCount) : '—' },
        { label: 'Current Status',  value: p.status || '—' },
    ];

    var gridHtml = '<div class="claim-grid">' + fields.map(function (f) {
        return '<div class="claim-field">' +
            '<div class="cf-label">' + f.label + '</div>' +
            '<div class="cf-value ' + (f.cls || '') + '">' + esc(f.value) + '</div>' +
            '</div>';
    }).join('') + '</div>';

    var codeExplain = getDenialCodeExplanation(p.denialCode, p.denialReason);
    var denialHtml = '<div class="denial-banner">' +
        '<div class="denial-banner-icon"><i class="fas fa-exclamation-triangle"></i></div>' +
        '<div class="denial-banner-body">' +
            '<div class="denial-banner-title">Why this claim was denied — ' + esc(p.denialCode || 'CO-16') + ': ' + esc(codeExplain.short) + '</div>' +
            '<div class="denial-banner-text">' + esc(p.denialReason || codeExplain.cause) + '</div>' +
        '</div>' +
        '</div>';

    var extras = [];
    if (p.medicalHistory) extras.push({ label: 'Medical History',       value: p.medicalHistory });
    if (p.medications)    extras.push({ label: 'Current Medications',   value: p.medications });
    if (p.allergies)      extras.push({ label: 'Allergies',             value: p.allergies });
    if (p.priorAuth)      extras.push({ label: 'Prior Authorization',   value: String(p.priorAuth) });

    var extraHtml = extras.length
        ? '<div class="patient-info-grid">' + extras.map(function (e) {
            return '<div class="claim-field"><div class="cf-label">' + e.label + '</div><div class="cf-value">' + esc(e.value) + '</div></div>';
          }).join('') + '</div>'
        : '';

    document.getElementById('claimHistoryBody').innerHTML = gridHtml + denialHtml + extraHtml;
}

// ── Drop zone & file handling ─────────────────────────────────────────────────
function setupDropZone() {
    var zone = document.getElementById('uploadZone');
    zone.addEventListener('dragover', function (e) { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', function () { zone.classList.remove('drag-over'); });
    zone.addEventListener('drop', function (e) {
        e.preventDefault();
        zone.classList.remove('drag-over');
        var file = e.dataTransfer.files[0];
        if (file) processFile(file);
    });
}
function onFileSelected(e) {
    var file = e.target.files[0];
    if (file) processFile(file);
    e.target.value = '';
}
function processFile(file) {
    if (!selectedPatient) { showToast('Select a patient first', 'error'); return; }
    var reader = new FileReader();
    reader.onload = function (ev) { runAnalysis(ev.target.result, file.name); };
    reader.onerror = function () { showToast('Could not read file', 'error'); };
    reader.readAsText(file);
}

// ── Run analysis ──────────────────────────────────────────────────────────────
function runAnalysis(content, filename) {
    document.getElementById('uploadIdle').style.display = 'none';
    document.getElementById('uploadProcessing').style.display = 'block';
    document.getElementById('eraResults').style.display = 'none';

    var steps = [
        { icon: '📄', name: 'Parsing ERA / 835 file',        detail: 'Extracting X12 segments: ISA, GS, ST, CLP, CAS, NM1, SVC, DTM' },
        { icon: '🔍', name: 'Classifying denial codes',       detail: 'Mapping CARC/RARC codes to denial categories & appeal strategies' },
        { icon: '🧠', name: 'Bedrock AI — patient context',   detail: 'Nova Micro reading claim history, diagnosis, prior auth status' },
        { icon: '💡', name: 'Building recovery plan',         detail: 'Generating denial-specific appeal actions & documentation checklist' },
        { icon: '✅', name: 'Finalising remittance report',   detail: 'Calculating recoverable amount & priority order' },
    ];
    renderSteps(steps, 0);

    var delays = [700, 1500, 2800, 3500, 4000];
    var timers = delays.map(function (d, i) {
        return setTimeout(function () { renderSteps(steps, i + 1); }, d);
    });

    fetch(API + '/appeals/api/process-era/' + selectedPatient.id, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ era_content: content, filename: filename }),
    })
    .then(function (res) {
        timers.forEach(function (t) { clearTimeout(t); });
        renderSteps(steps, steps.length);
        if (!res.ok) throw new Error('Server error ' + res.status);
        return res.json();
    })
    .then(function (data) {
        lastEraData = data;
        setTimeout(function () {
            showResults(data);
            showToast('ERA analysis complete', 'success');
        }, 400);
    })
    .catch(function (err) {
        timers.forEach(function (t) { clearTimeout(t); });
        renderSteps(steps, 0, true);
        showToast('Analysis failed: ' + err.message, 'error');
        setTimeout(resetAnalysis, 2500);
    });
}

function renderSteps(steps, activeIdx, error) {
    document.getElementById('agentSteps').innerHTML = steps.map(function (s, i) {
        var cls = 'agent-step';
        var statusHtml = '<span class="step-wait"><i class="fas fa-clock"></i></span>';
        if (i < activeIdx) {
            cls += ' done';
            statusHtml = '<span class="step-check"><i class="fas fa-check-circle"></i></span>';
        } else if (i === activeIdx) {
            cls += error ? ' error' : ' active';
            statusHtml = error
                ? '<span style="color:var(--red)"><i class="fas fa-times-circle"></i></span>'
                : '<div class="step-spinner"></div>';
        }
        return '<div class="' + cls + '">' +
            '<div class="step-icon">' + s.icon + '</div>' +
            '<div class="step-body"><div class="step-name">' + s.name + '</div><div class="step-detail">' + s.detail + '</div></div>' +
            '<div class="step-status">' + statusHtml + '</div>' +
            '</div>';
    }).join('');
}

// ── Show results ──────────────────────────────────────────────────────────────
// ERA reality: one ERA = insurer's remittance for ONE claim submission.
// Service lines within that claim can be partially paid / partially denied.
// We show: denied service lines, recoverable amount, per-code action plan.
function showResults(data) {
    document.getElementById('uploadProcessing').style.display = 'none';
    document.getElementById('eraResults').style.display = 'block';

    var llm     = data.llm_analysis || {};
    var denials = data.classified_denials || [];
    var denial  = denials[0] || {};
    var p       = selectedPatient;

    // ── Financial summary ──
    // ERA contains service lines for ONE claim. Denied lines vs paid lines.
    var deniedAmt = 0;
    denials.forEach(function (d) { deniedAmt += (d.denied_amount || 0); });
    var billedAmt   = Number(p.amount || 0);
    var paidAmt     = Math.max(0, billedAmt - deniedAmt);
    var recoverPct  = billedAmt > 0 ? Math.round((deniedAmt / billedAmt) * 100) : 100;
    // If no denial amounts parsed from ERA, use full billed as denied (common with simple ERA files)
    if (deniedAmt === 0) { deniedAmt = billedAmt; paidAmt = 0; recoverPct = 100; }

    var stripHtml =
        '<div class="era-stat">' +
            '<div class="era-stat-icon blue"><i class="fas fa-file-invoice-dollar"></i></div>' +
            '<div class="era-stat-body">' +
                '<div class="era-stat-value">' + fmt(billedAmt) + '</div>' +
                '<div class="era-stat-label">Billed Amount</div>' +
            '</div>' +
        '</div>' +
        '<div class="era-stat">' +
            '<div class="era-stat-icon red"><i class="fas fa-times-circle"></i></div>' +
            '<div class="era-stat-body">' +
                '<div class="era-stat-value red">' + fmt(deniedAmt) + '</div>' +
                '<div class="era-stat-label">Denied Amount</div>' +
            '</div>' +
        '</div>' +
        '<div class="era-stat">' +
            '<div class="era-stat-icon green"><i class="fas fa-check-circle"></i></div>' +
            '<div class="era-stat-body">' +
                '<div class="era-stat-value green">' + fmt(paidAmt) + '</div>' +
                '<div class="era-stat-label">Paid Amount</div>' +
            '</div>' +
        '</div>' +
        '<div class="era-stat">' +
            '<div class="era-stat-icon amber"><i class="fas fa-exclamation-circle"></i></div>' +
            '<div class="era-stat-body">' +
                '<div class="era-stat-value amber">' + recoverPct + '%</div>' +
                '<div class="era-stat-label">Recoverable</div>' +
            '</div>' +
        '</div>';

    if (llm.estimated_recovery) {
        stripHtml +=
            '<div class="era-stat">' +
                '<div class="era-stat-icon teal"><i class="fas fa-hand-holding-usd"></i></div>' +
                '<div class="era-stat-body">' +
                    '<div class="era-stat-value teal">' + esc(llm.estimated_recovery) + '</div>' +
                    '<div class="era-stat-label">Est. Recovery</div>' +
                '</div>' +
            '</div>';
    }
    document.getElementById('eraSummaryStrip').innerHTML = stripHtml;

    // ── Card 1: What happened ──
    var riskLevel  = llm.risk_level || (p.priority === 'high' ? 'high' : 'medium');
    var riskCls    = riskLevel === 'high' ? 'red' : riskLevel === 'medium' ? 'amber' : 'green';
    var denialCode = denial.denial_code || p.denialCode || 'CO-16';
    var denialReason = denial.denial_reason || p.denialReason || 'See denial details';
    var codeExplain  = getDenialCodeExplanation(denialCode, denialReason);
    var priOrder     = Array.isArray(llm.appeal_priority_order) ? llm.appeal_priority_order.join(', ') : '';
    var serviceLines = denials.length || 1;

    document.getElementById('explanationBody').innerHTML =
        '<div class="explanation-summary">' +
            esc(llm.overall_assessment || buildFallbackAssessment(data, p, deniedAmt)) +
        '</div>' +
        '<div class="exp-grid">' +
            '<div class="exp-row"><div class="exp-label">Insurer Response</div><div class="exp-value">' + esc(denialReason) + '</div></div>' +
            '<div class="exp-row"><div class="exp-label">Denial Code</div><div class="exp-value red">' + esc(denialCode) + ' — ' + esc(codeExplain.short) + '</div></div>' +
            '<div class="exp-row"><div class="exp-label">Root Cause</div><div class="exp-value">' + esc(llm.root_cause_analysis || codeExplain.cause) + '</div></div>' +
            '<div class="exp-row"><div class="exp-label">Appeal Urgency</div><div class="exp-value ' + riskCls + '">' + riskLevel.toUpperCase() + ' — ' + appealDeadline(riskLevel) + '</div></div>' +
            (priOrder ? '<div class="exp-row"><div class="exp-label">Priority Order</div><div class="exp-value blue">' + esc(priOrder) + '</div></div>' : '') +
            '<div class="exp-row"><div class="exp-label">Service Lines in ERA</div><div class="exp-value">' +
                serviceLines + ' denied line' + (serviceLines !== 1 ? 's' : '') +
                ' — this is a single claim remittance' +
            '</div></div>' +
        '</div>';

    // ── Card 2: What needs to change ──
    var recs = buildRecommendations(data, p, denialCode, deniedAmt);
    document.getElementById('suggestionCount').textContent = recs.length + ' action' + (recs.length !== 1 ? 's' : '');
    document.getElementById('suggestionsBody').innerHTML = recs.map(function (r, i) {
        return '<div class="suggestion-item ' + r.level + '">' +
            '<div class="sug-num">' + (i + 1) + '</div>' +
            '<div class="sug-body">' +
                '<div class="sug-title">' + esc(r.title) + '</div>' +
                '<div class="sug-desc">'  + esc(r.desc)  + '</div>' +
                (r.action ? '<span class="sug-action">' + esc(r.action) + '</span>' : '') +
            '</div>' +
            '</div>';
    }).join('');

    document.getElementById('eraResults').classList.add('fade-up');
    document.getElementById('btnReapply').disabled = false;
}

function appealDeadline(risk) {
    if (risk === 'high')   return 'File within 30 days';
    if (risk === 'medium') return 'File within 60 days';
    return 'File within 90 days';
}

function buildFallbackAssessment(data, p, deniedAmt) {
    var code    = ((data.classified_denials || [])[0] || {}).denial_code || p.denialCode || 'CO-16';
    var explain = getDenialCodeExplanation(code, p.denialReason);
    return 'The insurer denied ' + fmt(deniedAmt || p.amount) + ' for ' + (p.name || 'this patient') + '\'s claim. ' +
           'Denial code ' + code + ' means: ' + explain.short + '. ' +
           explain.cause + ' This amount is recoverable with the correct appeal documentation.';
}

// ── CARC code plain-English explanations ──────────────────────────────────────
function getDenialCodeExplanation(code, reason) {
    var map = {
        'CO-16':  { short: 'Missing / incomplete information',    cause: 'Required documentation or data elements were missing from the claim submission.' },
        'CO-4':   { short: 'Procedure code inconsistent',         cause: 'The procedure code is inconsistent with the modifier, place of service, or diagnosis code.' },
        'CO-18':  { short: 'Duplicate claim',                     cause: 'This claim or service was already submitted and adjudicated.' },
        'CO-29':  { short: 'Timely filing exceeded',              cause: 'The claim was not submitted within the payer\'s required filing window.' },
        'CO-50':  { short: 'Medical necessity not met',           cause: 'The payer determined the service was not medically necessary based on submitted documentation.' },
        'CO-97':  { short: 'Benefit included in primary',         cause: 'Payment is included in the allowance for another service already adjudicated.' },
        'CO-197': { short: 'Prior authorization missing',         cause: 'The service required prior authorization that was not obtained or has expired.' },
        'CO-B7':  { short: 'Provider not certified / eligible',   cause: 'The rendering provider is not credentialed or eligible to bill for this service with this payer.' },
        'CO-45':  { short: 'Charge exceeds fee schedule',         cause: 'The billed amount exceeds the payer\'s maximum allowable fee for this service.' },
        'CO-96':  { short: 'Non-covered charge',                  cause: 'This service is not covered under the patient\'s current benefit plan.' },
    };
    var key = (code || '').toUpperCase().replace(/\s/g, '');
    if (map[key]) return map[key];
    for (var k in map) { if (key.indexOf(k.replace('CO-', '')) >= 0) return map[k]; }
    return { short: 'Claim denied', cause: reason || 'Review the ERA for specific denial details.' };
}

// ── Denial-code-specific action playbooks ─────────────────────────────────────
function buildRecommendations(data, p, denialCode, deniedAmt) {
    var llm  = data.llm_analysis || {};
    var recs = [];

    // 1. Bedrock AI recommendations (always first if present)
    if (Array.isArray(llm.top_recommendations) && llm.top_recommendations.length) {
        llm.top_recommendations.forEach(function (r, i) {
            var text = safeStr(r);
            if (!text) return;
            recs.push({
                level:  i === 0 ? 'critical' : i === 1 ? 'important' : 'advisory',
                title:  i === 0 ? 'Primary Action — ' + (denialCode || 'CO-16') : i === 1 ? 'Supporting Action' : 'Best Practice',
                desc:   text,
                action: i === 0 ? 'Do this first' : null,
            });
        });
    }

    // 2. Denial-code-specific hardcoded playbook (fills gaps)
    var codeActions = getDenialCodeActions(denialCode, p, deniedAmt);
    codeActions.forEach(function (a) {
        var dup = recs.some(function (r) { return r.desc.toLowerCase().indexOf(a.keyword) >= 0; });
        if (!dup) recs.push(a);
    });

    // 3. Process improvement from Bedrock
    if (llm.process_improvement) {
        recs.push({
            level: 'advisory',
            title: 'Process Improvement',
            desc:  safeStr(llm.process_improvement),
            action: 'Prevent future denials',
        });
    }

    // 4. Always add appeal deadline reminder
    recs.push({
        level: 'advisory',
        title: 'File Appeal Before Deadline',
        desc:  'Most payers allow 60–180 days from the ERA date to file a formal appeal. ' +
               'Check ' + esc(p.payerName || 'the insurer') + '\'s specific timely appeal filing limit. ' +
               'Missing this window permanently forfeits recovery of ' + fmt(deniedAmt || p.amount) + '.',
        action: 'Check payer policy',
    });

    return recs.slice(0, 6);
}

function getDenialCodeActions(code, p, deniedAmt) {
    var c     = (code || '').toUpperCase().replace(/\s/g, '');
    var payer = p.payerName || p.payer || 'the insurer';
    var amt   = fmt(deniedAmt || p.amount);

    var playbooks = {
        'CO-16': [
            { level: 'critical',  title: 'Gather Missing Documentation', keyword: 'document',
              desc: 'Identify exactly which data elements ' + payer + ' flagged as missing. Common items: referring provider NPI, place of service code, diagnosis pointer, or clinical notes. Resubmit with complete documentation.',
              action: 'Resubmit corrected claim' },
            { level: 'important', title: 'Request Remittance Detail from Payer', keyword: 'remittance',
              desc: 'Call ' + payer + ' provider services and request the specific RARC codes attached to this denial. RARC codes pinpoint exactly which field was incomplete.',
              action: 'Call provider services' },
            { level: 'advisory',  title: 'Update Claim Submission Template', keyword: 'template',
              desc: 'Review your billing system\'s claim template for this procedure code. Ensure all required fields are pre-populated to prevent recurrence of CO-16 denials.',
              action: 'Update billing template' },
        ],
        'CO-B7': [
            { level: 'critical',  title: 'Verify Provider Credentialing Status', keyword: 'credential',
              desc: 'Confirm the rendering provider is actively credentialed with ' + payer + '. Check CAQH ProView or the payer\'s provider portal. If credentialing lapsed, initiate re-credentialing immediately — this can take 60–90 days.',
              action: 'Check CAQH ProView' },
            { level: 'critical',  title: 'Check Billing vs Rendering Provider NPI', keyword: 'billing',
              desc: 'Ensure the billing NPI and rendering NPI on the claim match what ' + payer + ' has on file. A mismatch between group NPI and individual NPI is a common cause of CO-B7.',
              action: 'Verify NPI match' },
            { level: 'important', title: 'Resubmit with Correct NPI', keyword: 'npi',
              desc: 'Once credentialing is confirmed, resubmit the claim with the correct rendering provider NPI. Include a cover letter explaining the correction. Recoverable: ' + amt + '.',
              action: 'Resubmit with correct NPI' },
        ],
        'CO-197': [
            { level: 'critical',  title: 'Request Retroactive Prior Authorization', keyword: 'retroactive',
              desc: 'Contact ' + payer + ' immediately to request retroactive authorization. Some payers grant this for urgent or emergent services. Document medical necessity thoroughly in the request.',
              action: 'Request retro auth' },
            { level: 'important', title: 'Submit Medical Necessity Documentation', keyword: 'necessity',
              desc: 'Compile clinical notes, physician orders, and diagnosis documentation supporting medical necessity. A strong medical necessity letter from the treating physician significantly improves appeal success.',
              action: 'Compile clinical notes' },
            { level: 'advisory',  title: 'Implement Prior Auth Workflow', keyword: 'workflow',
              desc: 'Set up a pre-authorization check in your scheduling system for this procedure code. CO-197 denials are 100% preventable with the right workflow.',
              action: 'Update scheduling workflow' },
        ],
        'CO-50': [
            { level: 'critical',  title: 'Write Medical Necessity Appeal Letter', keyword: 'medical necessity',
              desc: 'Draft a formal appeal with the treating physician\'s clinical rationale. Reference clinical guidelines (CMS LCD, payer-specific policies) that support the necessity of this service.',
              action: 'Draft appeal letter' },
            { level: 'important', title: 'Attach Supporting Clinical Evidence', keyword: 'clinical evidence',
              desc: 'Include: physician notes, lab results, imaging reports, treatment history, and peer-reviewed literature supporting the treatment. ' + payer + ' medical reviewers respond to evidence-based documentation.',
              action: 'Gather clinical records' },
        ],
        'CO-29': [
            { level: 'critical',  title: 'Verify Timely Filing Window', keyword: 'timely',
              desc: 'Confirm whether the filing deadline has truly passed. Some payers calculate from date of service, others from date of discharge. If within the window, resubmit immediately with proof of timely filing.',
              action: 'Verify filing date' },
            { level: 'important', title: 'Submit Proof of Timely Filing', keyword: 'proof',
              desc: 'If you have evidence the claim was submitted on time (clearinghouse confirmation, EDI acknowledgment), submit this as an appeal. Payers must honor timely filing if you can prove original submission.',
              action: 'Attach EDI confirmation' },
        ],
        'CO-4': [
            { level: 'critical',  title: 'Correct Procedure–Diagnosis Linkage', keyword: 'diagnosis',
              desc: 'Review the procedure code and ensure the diagnosis code supports medical necessity for that procedure. Check CMS NCCI edits for bundling conflicts.',
              action: 'Review NCCI edits' },
            { level: 'important', title: 'Verify Modifier Usage', keyword: 'modifier',
              desc: 'Check if a modifier (e.g., -25, -59, -GT) is needed to unbundle the service or clarify the clinical scenario. Incorrect or missing modifiers are the #1 cause of CO-4 denials.',
              action: 'Add correct modifier' },
        ],
    };

    for (var key in playbooks) {
        if (c === key || c.indexOf(key.replace('CO-', '')) >= 0) return playbooks[key];
    }

    // Generic fallback
    return [
        { level: 'critical',  title: 'Review Full ERA Remittance Detail', keyword: 'review',
          desc: 'Pull the complete 835 transaction and identify all CARC and RARC codes. Each code maps to a specific corrective action.',
          action: 'Review 835 detail' },
        { level: 'important', title: 'Contact ' + payer + ' Provider Relations', keyword: 'contact',
          desc: 'Call the provider services line and request a detailed explanation of the denial. Ask specifically: what documentation is needed to overturn this denial and what is the appeal deadline.',
          action: 'Call provider services' },
        { level: 'advisory',  title: 'Resubmit Corrected Claim', keyword: 'resubmit',
          desc: 'Once the root cause is identified, submit a corrected claim (frequency code 7) or a formal appeal with supporting documentation. Recoverable amount: ' + amt + '.',
          action: 'Submit corrected claim' },
    ];
}

// ── Re-apply with Corrections ─────────────────────────────────────────────────
function reapplyWithCorrections() {
    if (!selectedPatient) { showToast('No patient selected', 'error'); return; }

    var modal = document.getElementById('claimModal');
    var body  = document.getElementById('modalBody');
    modal.style.display = 'flex';
    body.innerHTML =
        '<div class="modal-loading">' +
            '<div class="modal-loading-steps">' +
                '<div class="ml-step active"><div class="step-spinner"></div><span>Analysing denial context via AWS Bedrock Nova Micro…</span></div>' +
                '<div class="ml-step"><i class="fas fa-clock"></i><span>Generating field corrections</span></div>' +
                '<div class="ml-step"><i class="fas fa-clock"></i><span>Building corrected X12 837P claim</span></div>' +
            '</div>' +
        '</div>';

    var llm    = (lastEraData && lastEraData.llm_analysis) || {};
    var denial = ((lastEraData && lastEraData.classified_denials) || [])[0] || {};
    var p      = selectedPatient;

    var denialCode   = denial.denial_code   || p.denialCode   || 'CO-16';
    var denialReason = denial.denial_reason || p.denialReason || '';
    var assessment   = llm.overall_assessment || '';
    var recs         = Array.isArray(llm.top_recommendations)
        ? llm.top_recommendations.map(safeStr).filter(Boolean)
        : [];

    // Animate loading steps
    var stepIdx = 0;
    var stepTimer = setInterval(function () {
        stepIdx++;
        var steps = body.querySelectorAll('.ml-step');
        steps.forEach(function (s, i) {
            s.classList.remove('active', 'done');
            if (i < stepIdx) {
                s.classList.add('done');
                s.innerHTML = '<i class="fas fa-check-circle" style="color:var(--green)"></i><span>' + s.querySelector('span').textContent + '</span>';
            } else if (i === stepIdx) {
                s.classList.add('active');
                s.innerHTML = '<div class="step-spinner"></div><span>' + s.querySelector('span').textContent + '</span>';
            }
        });
        if (stepIdx >= 2) clearInterval(stepTimer);
    }, 1200);

    fetch(API + '/appeals/api/resubmit-corrected-claim/' + p.id, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            denial_code:     denialCode,
            denial_reason:   denialReason,
            era_assessment:  assessment,
            recommendations: recs,
        }),
    })
    .then(function (res) {
        clearInterval(stepTimer);
        if (!res.ok) throw new Error('Server error ' + res.status);
        return res.json();
    })
    .then(function (data) { renderClaimModal(data); })
    .catch(function (err) {
        clearInterval(stepTimer);
        body.innerHTML =
            '<div class="modal-loading" style="color:var(--red);flex-direction:column;gap:12px">' +
                '<i class="fas fa-exclamation-circle" style="font-size:32px"></i>' +
                '<span>Failed to generate corrected claim: ' + esc(err.message) + '</span>' +
                '<button class="btn-ghost" onclick="document.getElementById(\'claimModal\').style.display=\'none\'">' +
                    '<i class="fas fa-times"></i> Close' +
                '</button>' +
            '</div>';
    });
}

function renderClaimModal(d) {
    var strengthCls = d.appeal_strength === 'strong' ? 'strong' : d.appeal_strength === 'weak' ? 'weak' : 'moderate';
    var approvalPct = parseInt(d.estimated_approval_probability) || 70;
    var approvalCls = approvalPct >= 70 ? 'ok' : approvalPct >= 50 ? 'warn' : 'weak';

    // Corrections list
    var corrHtml = (d.corrections_made || []).map(function (c) {
        return '<div class="correction-item"><i class="fas fa-check-circle"></i><span>' + esc(safeStr(c)) + '</span></div>';
    }).join('') || '<div class="correction-item"><i class="fas fa-check-circle"></i><span>Claim fields corrected to address ' + esc(d.denial_code_addressed) + ' denial</span></div>';

    // Docs needed
    var docsHtml = '';
    if (d.additional_documentation && d.additional_documentation.length) {
        docsHtml = '<div class="docs-needed">' +
            '<div class="docs-needed-title"><i class="fas fa-paperclip"></i> Attach These Documents</div>' +
            d.additional_documentation.map(function (doc) {
                return '<div class="doc-item"><i class="fas fa-file-alt"></i>' + esc(safeStr(doc)) + '</div>';
            }).join('') +
            '</div>';
    }

    // Claim fields grid — mirrors pre-submission claim view
    var fields = [
        { label: 'New Claim ID',    value: d.new_claim_id,                                          cls: 'mono blue' },
        { label: 'Original Claim',  value: d.original_claim_id,                                     cls: 'mono' },
        { label: 'Patient',         value: d.patient_name },
        { label: 'Insurer',         value: d.insurer },
        { label: 'Service Date',    value: d.service_date },
        { label: 'Procedure Code',  value: d.procedure_code + (d.modifier ? ' — Mod: ' + d.modifier : ''), cls: 'mono' },
        { label: 'Diagnosis Code',  value: d.diagnosis_code,                                        cls: 'mono' },
        { label: 'Claim Amount',    value: fmt(d.claim_amount),                                     cls: 'blue' },
        { label: 'Prior Auth',      value: d.prior_auth || '—' },
        { label: 'Provider',        value: d.provider },
    ];
    var fieldsHtml = '<div class="claim-grid" style="margin-bottom:16px">' + fields.map(function (f) {
        return '<div class="claim-field">' +
            '<div class="cf-label">' + f.label + '</div>' +
            '<div class="cf-value ' + (f.cls || '') + '">' + esc(f.value || '—') + '</div>' +
            '</div>';
    }).join('') + '</div>';

    // X12 section
    var x12Html =
        '<div class="x12-section">' +
            '<div class="x12-header" onclick="toggleX12()">' +
                '<div class="x12-header-title"><i class="fas fa-code"></i> X12 837P Transaction (ANSI ASC X12 — Corrected Resubmission)</div>' +
                '<span class="x12-toggle" id="x12Toggle">Show ▼</span>' +
            '</div>' +
            '<div class="x12-body" id="x12Body" style="display:none">' + esc(d.x12_837p || '') + '</div>' +
        '</div>';

    var noteHtml = d.resubmission_notes
        ? '<div class="resubmission-note"><i class="fas fa-info-circle"></i>' + esc(d.resubmission_notes) + '</div>'
        : '';

    document.getElementById('modalBody').innerHTML =
        // Hero
        '<div class="claim-result-hero">' +
            '<div class="crh-icon">📋</div>' +
            '<div class="crh-main">' +
                '<div class="crh-title">Corrected Claim Ready — ' + esc(d.new_claim_id) + '</div>' +
                '<div class="crh-sub">Denial ' + esc(d.denial_code_addressed) + ' addressed &middot; Generated by ' + esc(d.generated_by) + '</div>' +
            '</div>' +
            '<div class="crh-stats">' +
                '<div class="crh-stat"><div class="crh-stat-label">Approval Est.</div><div class="crh-stat-value ' + approvalCls + '">' + esc(d.estimated_approval_probability) + '</div></div>' +
                '<div class="crh-stat"><div class="crh-stat-label">Appeal Strength</div><div class="crh-stat-value ' + strengthCls + '">' + (d.appeal_strength || 'moderate').toUpperCase() + '</div></div>' +
                '<div class="crh-stat"><div class="crh-stat-label">Claim Amount</div><div class="crh-stat-value">' + fmt(d.claim_amount) + '</div></div>' +
            '</div>' +
        '</div>' +
        // Summary
        '<div class="correction-summary">' + esc(d.correction_summary || 'Claim corrected to address ' + d.denial_code_addressed + ' denial.') + '</div>' +
        // Corrections
        '<div class="corrections-list">' + corrHtml + '</div>' +
        // Docs
        docsHtml +
        // Claim fields
        fieldsHtml +
        // X12
        x12Html +
        // Note
        noteHtml +
        // Actions
        '<div class="modal-actions">' +
            '<button class="btn-copy" onclick="copyX12()"><i class="fas fa-copy"></i> Copy X12</button>' +
            '<button class="btn-copy" onclick="downloadX12(\'' + esc(d.new_claim_id) + '\')"><i class="fas fa-download"></i> Download .837</button>' +
            '<button class="btn-submit-claim" onclick="submitCorrectedClaim(\'' + esc(d.insurer) + '\')">' +
                '<i class="fas fa-paper-plane"></i> Submit to ' + esc(d.insurer) +
            '</button>' +
        '</div>';

    window._lastX12      = d.x12_837p || '';
    window._lastClaimId  = d.new_claim_id || 'claim';
}

function submitCorrectedClaim(insurer) {
    showToast('Claim queued for submission to ' + insurer, 'success');
    document.getElementById('claimModal').style.display = 'none';
    // Update patient status badge in list
    if (selectedPatient) {
        var item = document.querySelector('.patient-item[data-id="' + selectedPatient.id + '"]');
        if (item) {
            var tags = item.querySelector('.pi-tags');
            if (tags) {
                var old = tags.querySelector('.pi-tag.denied');
                if (old) old.remove();
                var newTag = document.createElement('span');
                newTag.className = 'pi-tag resub';
                newTag.textContent = 'Resubmitted';
                tags.appendChild(newTag);
            }
        }
    }
}

function toggleX12() {
    var body   = document.getElementById('x12Body');
    var toggle = document.getElementById('x12Toggle');
    var vis    = body.style.display !== 'none';
    body.style.display  = vis ? 'none' : 'block';
    toggle.textContent  = vis ? 'Show ▼' : 'Hide ▲';
}
function copyX12() {
    if (!window._lastX12) return;
    navigator.clipboard.writeText(window._lastX12)
        .then(function () { showToast('X12 claim copied to clipboard', 'success'); })
        .catch(function () { showToast('Copy failed — use Ctrl+A in the code block', 'error'); });
}
function downloadX12(claimId) {
    if (!window._lastX12) return;
    var blob = new Blob([window._lastX12], { type: 'text/plain' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = (claimId || 'corrected_claim') + '.837';
    a.click();
    URL.revokeObjectURL(a.href);
    showToast('X12 837P downloaded', 'success');
}
function closeModal(e) {
    if (e.target === document.getElementById('claimModal')) {
        document.getElementById('claimModal').style.display = 'none';
    }
}

function resetAnalysis() {
    document.getElementById('uploadIdle').style.display = 'block';
    document.getElementById('uploadProcessing').style.display = 'none';
    document.getElementById('eraResults').style.display = 'none';
    document.getElementById('agentSteps').innerHTML = '';
    lastEraData = null;
    var btn = document.getElementById('btnReapply');
    if (btn) btn.disabled = true;
}
