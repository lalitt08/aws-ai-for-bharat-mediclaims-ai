/* MediClaims AI — ERA Processing */
'use strict';
var API = '';
var patients = [];
var selectedPatient = null;

document.addEventListener('DOMContentLoaded', function() {
    loadPatients();
    setupSearch();
    setupDropZone();
    document.getElementById('eraFile').addEventListener('change', onFileSelected);
});

function loadPatients() {
    fetch(API + '/appeals/api/denied-claims')
        .then(function(r) { return r.json(); })
        .then(function(data) {
            patients = data.denied_claims || [];
            renderPatientList(patients);
            document.getElementById('patientCount').textContent = patients.length;
        })
        .catch(function() {
            showToast('Failed to load patients', 'error');
            document.getElementById('patientList').innerHTML =
                '<div class="list-loading" style="color:var(--red)"><i class="fas fa-exclamation-circle"></i><span>Could not load patients</span></div>';
        });
}

function renderPatientList(list) {
    var el = document.getElementById('patientList');
    if (!list.length) { el.innerHTML = '<div class="list-loading"><span>No patients found</span></div>'; return; }
    el.innerHTML = list.map(function(p) {
        var ini = initials2(p.name);
        var pri = p.priority || 'medium';
        var amt = p.amount ? ('$' + Number(p.amount).toLocaleString()) : '';
        var stag = statusBadge(p.status);
        return '<div class="patient-item" data-id="' + p.id + '" onclick="selectPatient(\'' + p.id + '\')">' +
            '<div class="pi-avatar">' + ini + '</div>' +
            '<div class="pi-info">' +
                '<div class="pi-name">' + esc(p.name) + '</div>' +
                '<div class="pi-sub">' + esc(p.claimId || p.id) + '</div>' +
                '<div class="pi-tags"><span class="pi-tag ' + pri + '">' + pri + '</span>' + stag + '</div>' +
                (amt ? '<div class="pi-amount">' + amt + ' &middot; ' + esc(p.payerName || p.payer || '') + '</div>' : '') +
            '</div></div>';
    }).join('');
}

function statusBadge(status) {
    if (!status) return '';
    var s = status.toLowerCase();
    if (s.indexOf('resub') >= 0)  return '<span class="pi-tag resub">Resubmitted</span>';
    if (s.indexOf('appeal') >= 0) return '<span class="pi-tag appeal">Appeal</span>';
    return '<span class="pi-tag denied">Denied</span>';
}

function setupSearch() {
    document.getElementById('searchInput').addEventListener('input', function(e) {
        var q = e.target.value.toLowerCase();
        renderPatientList(patients.filter(function(p) {
            return (p.name||'').toLowerCase().indexOf(q)>=0 ||
                   (p.claimId||'').toLowerCase().indexOf(q)>=0 ||
                   (p.payerName||'').toLowerCase().indexOf(q)>=0 ||
                   (p.denialReason||'').toLowerCase().indexOf(q)>=0;
        }));
    });
}

function selectPatient(id) {
    selectedPatient = null;
    for (var i = 0; i < patients.length; i++) {
        if (patients[i].id === id) { selectedPatient = patients[i]; break; }
    }
    if (!selectedPatient) return;
    document.querySelectorAll('.patient-item').forEach(function(el) {
        el.classList.toggle('active', el.dataset.id === id);
    });
    document.getElementById('noSelection').style.display = 'none';
    document.getElementById('patientView').style.display = 'block';
    renderPatientHero(selectedPatient);
    renderClaimHistory(selectedPatient);
    resetAnalysis();
    document.getElementById('bedrockBadge').style.display = 'inline-flex';
}

function renderPatientHero(p) {
    var riskPct = p.riskScore ? Math.round(p.riskScore * 100) : 0;
    var riskCls = riskPct >= 60 ? 'danger' : riskPct >= 30 ? 'warn' : 'ok';
    var sucCls  = (p.successProbability||0) >= 60 ? 'ok' : (p.successProbability||0) >= 40 ? 'warn' : 'danger';
    document.getElementById('patientHero').innerHTML =
        '<div class="ph-avatar">' + initials2(p.name) + '</div>' +
        '<div class="ph-main">' +
            '<div class="ph-name">' + esc(p.name) + '</div>' +
            '<div class="ph-meta">' +
                '<span><i class="fas fa-birthday-cake"></i> Age ' + (p.age||'&mdash;') + '</span>' +
                '<span><i class="fas fa-hospital"></i> ' + esc(p.payerName||p.payer||'&mdash;') + '</span>' +
                '<span><i class="fas fa-user-md"></i> ' + esc(p.doctorName||'&mdash;') + '</span>' +
                '<span><i class="fas fa-calendar-alt"></i> ' + esc(p.serviceDate||'&mdash;') + '</span>' +
                '<span><i class="fas fa-id-card"></i> ' + esc(p.id) + '</span>' +
            '</div>' +
        '</div>' +
        '<div class="ph-stats">' +
            '<div class="ph-stat"><div class="ph-stat-label">Claim Amount</div><div class="ph-stat-value">$' + Number(p.amount||0).toLocaleString() + '</div></div>' +
            '<div class="ph-stat"><div class="ph-stat-label">Risk Score</div><div class="ph-stat-value ' + riskCls + '">' + riskPct + '%</div></div>' +
            '<div class="ph-stat"><div class="ph-stat-label">Appeal Success</div><div class="ph-stat-value ' + sucCls + '">' + (p.successProbability||0) + '%</div></div>' +
            '<div class="ph-stat"><div class="ph-stat-label">Denial Code</div><div class="ph-stat-value danger">' + esc(p.denialCode||'&mdash;') + '</div></div>' +
        '</div>';
}

function renderClaimHistory(p) {
    var s = (p.status||'').toLowerCase();
    var badgeCls = 'denied', badgeTxt = 'Denied';
    if (s.indexOf('resub') >= 0)    { badgeCls = 'resub';    badgeTxt = 'Resubmitted'; }
    else if (s.indexOf('appeal') >= 0) { badgeCls = 'appeal'; badgeTxt = 'Appeal Filed'; }
    else if (s.indexOf('approved') >= 0) { badgeCls = 'approved'; badgeTxt = 'Approved'; }
    var badge = document.getElementById('claimStatusBadge');
    badge.className = 'section-badge ' + badgeCls;
    badge.textContent = badgeTxt;

    var fields = [
        { label:'Claim ID',     value: p.claimId||'&mdash;',                      cls:'mono blue' },
        { label:'Service Date', value: p.serviceDate||'&mdash;' },
        { label:'Procedure',    value: p.procedure||'&mdash;',                     cls:'mono' },
        { label:'Insurer',      value: p.payerName||p.payer||'&mdash;' },
        { label:'Provider',     value: p.doctorName||'&mdash;' },
        { label:'Claim Amount', value: '$'+Number(p.amount||0).toLocaleString(),   cls:'blue' },
        { label:'Denial Code',  value: p.denialCode||'&mdash;',                    cls:'red' },
        { label:'Category',     value: p.denialCategory||'&mdash;' },
        { label:'Issues Found', value: p.issuesCount!=null ? String(p.issuesCount) : '&mdash;' },
        { label:'Status',       value: p.status||'&mdash;' },
    ];
    var gridHtml = '<div class="claim-grid">' + fields.map(function(f) {
        return '<div class="claim-field"><div class="cf-label">'+f.label+'</div><div class="cf-value '+(f.cls||'")>'+f.value+'</div></div>';
    }).join('') + '</div>';

    var denialHtml = '<div class="denial-banner">' +
        '<i class="fas fa-exclamation-triangle"></i>' +
        '<div class="denial-banner-body">' +
            '<div class="denial-banner-title">Denial Reason &mdash; ' + esc(p.denialCode||'CO-16') + '</div>' +
            '<div class="denial-banner-text">' + esc(p.denialReason||'No denial reason recorded.') + '</div>' +
        '</div></div>';

    var extras = [];
    if (p.medicalHistory) extras.push({label:'Medical History', value:p.medicalHistory});
    if (p.medications)    extras.push({label:'Medications',     value:p.medications});
    if (p.allergies)      extras.push({label:'Allergies',       value:p.allergies});
    if (p.priorAuth)      extras.push({label:'Prior Auth',      value:String(p.priorAuth)});
    var extraHtml = extras.length ? '<div class="patient-info-grid">' + extras.map(function(e) {
        return '<div class="claim-field"><div class="cf-label">'+e.label+'</div><div class="cf-value">'+esc(e.value)+'</div></div>';
    }).join('') + '</div>' : '';

    document.getElementById('claimHistoryBody').innerHTML = gridHtml + denialHtml + extraHtml;
}

function setupDropZone() {
    var zone = document.getElementById('uploadZone');
    zone.addEventListener('dragover', function(e) { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', function() { zone.classList.remove('drag-over'); });
    zone.addEventListener('drop', function(e) {
        e.preventDefault(); zone.classList.remove('drag-over');
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
    reader.onload = function(e) { runAnalysis(e.target.result, file.name); };
    reader.onerror = function() { showToast('Could not read file', 'error'); };
    reader.readAsText(file);
}

function runAnalysis(content, filename) {
    document.getElementById('uploadIdle').style.display = 'none';
    document.getElementById('uploadProcessing').style.display = 'block';
    document.getElementById('eraResults').style.display = 'none';

    var steps = [
        { icon:'📄', name:'Parsing ERA / 835 file',    detail:'Reading X12 segments: CLP, CAS, NM1, SVC, DTM' },
        { icon:'🔍', name:'Classifying denial codes',   detail:'Mapping CO codes to denial categories' },
        { icon:'🤖', name:'AI agent analysis',          detail:'AWS Bedrock Nova Micro — reading patient context' },
        { icon:'💡', name:'Generating recommendations', detail:'Building actionable suggestions' },
        { icon:'✅', name:'Finalising report',          detail:'Preparing natural language explanation' },
    ];
    renderSteps(steps, 0);

    var stepIdx = 0;
    var delays = [700, 1400, 2600, 3200, 3800];
    var timers = delays.map(function(d, i) {
        return setTimeout(function() { renderSteps(steps, i + 1); }, d);
    });

    fetch(API + '/appeals/api/process-era/' + selectedPatient.id, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ era_content: content, filename: filename }),
    })
    .then(function(res) {
        timers.forEach(function(t) { clearTimeout(t); });
        renderSteps(steps, steps.length);
        if (!res.ok) throw new Error('Server error ' + res.status);
        return res.json();
    })
    .then(function(data) {
        setTimeout(function() {
            showResults(data);
            showToast('ERA analysis complete', 'success');
        }, 400);
    })
    .catch(function(err) {
        timers.forEach(function(t) { clearTimeout(t); });
        renderSteps(steps, stepIdx, true);
        showToast('Analysis failed: ' + err.message, 'error');
        setTimeout(resetAnalysis, 2500);
    });
}

function renderSteps(steps, activeIdx, error) {
    document.getElementById('agentSteps').innerHTML = steps.map(function(s, i) {
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
            '<div class="step-status">' + statusHtml + '</div></div>';
    }).join('');
}

function showResults(data) {
    document.getElementById('uploadProcessing').style.display = 'none';
    document.getElementById('eraResults').style.display = 'block';

    var llm     = data.llm_analysis || {};
    var denial  = (data.classified_denials || [])[0] || {};
    var p       = selectedPatient;
    var summary = data.summary || {};

    // ERA summary strip
    var stripHtml =
        '<div class="era-stat"><div class="era-stat-value blue">'  + (summary.total_claims||0) + '</div><div class="era-stat-label">Total Claims</div></div>' +
        '<div class="era-stat"><div class="era-stat-value red">'   + (data.denials_count||0)   + '</div><div class="era-stat-label">Denied</div></div>' +
        '<div class="era-stat"><div class="era-stat-value green">' + (summary.paid_claims||0)  + '</div><div class="era-stat-label">Paid</div></div>';
    if (llm.estimated_recovery) {
        stripHtml += '<div class="era-stat"><div class="era-stat-value green">' + esc(llm.estimated_recovery) + '</div><div class="era-stat-label">Est. Recovery</div></div>';
    }
    document.getElementById('eraSummaryStrip').innerHTML = stripHtml;

    // Card 1: What happened
    var riskLevel = llm.risk_level || (p.priority === 'high' ? 'high' : 'medium');
    var riskCls   = riskLevel === 'high' ? 'red' : riskLevel === 'medium' ? 'amber' : 'green';
    var priOrder  = Array.isArray(llm.appeal_priority_order) ? llm.appeal_priority_order.join(', ') : '';

    document.getElementById('explanationBody').innerHTML =
        '<div class="explanation-summary">' + esc(llm.overall_assessment || buildFallbackAssessment(data, p)) + '</div>' +
        '<div class="exp-grid">' +
            '<div class="exp-row"><div class="exp-label">Root Cause</div><div class="exp-value">' + esc(llm.root_cause_analysis || denial.denial_reason || p.denialReason || '&mdash;') + '</div></div>' +
            '<div class="exp-row"><div class="exp-label">Denial Code</div><div class="exp-value red">' + esc(denial.denial_code || p.denialCode || '&mdash;') + '</div></div>' +
            '<div class="exp-row"><div class="exp-label">Denial Reason</div><div class="exp-value">' + esc(denial.denial_reason || p.denialReason || '&mdash;') + '</div></div>' +
            '<div class="exp-row"><div class="exp-label">Risk Level</div><div class="exp-value ' + riskCls + '">' + riskLevel.toUpperCase() + '</div></div>' +
            (priOrder ? '<div class="exp-row"><div class="exp-label">Appeal Priority</div><div class="exp-value blue">' + esc(priOrder) + '</div></div>' : '') +
        '</div>';

    // Card 2: What needs to change
    var recs = buildRecommendations(data, p);
    document.getElementById('suggestionCount').textContent = recs.length + ' suggestion' + (recs.length !== 1 ? 's' : '');
    document.getElementById('suggestionsBody').innerHTML = recs.map(function(r, i) {
        return '<div class="suggestion-item ' + r.level + '">' +
            '<div class="sug-num">' + (i+1) + '</div>' +
            '<div class="sug-body">' +
                '<div class="sug-title">' + esc(r.title) + '</div>' +
                '<div class="sug-desc">'  + esc(r.desc)  + '</div>' +
                (r.action ? '<span class="sug-action">' + esc(r.action) + '</span>' : '') +
            '</div></div>';
    }).join('');

    document.getElementById('eraResults').classList.add('fade-up');
}

function buildFallbackAssessment(data, p) {
    var denied = data.denials_count || 0;
    var total  = (data.summary && data.summary.total_claims) || 1;
    return 'This ERA contains ' + total + ' claim(s), of which ' + denied + ' were denied. The primary denial for ' + p.name + ' is related to ' + (p.denialReason||'documentation issues') + ' (' + (p.denialCode||'CO-16') + '). Immediate action is recommended to recover the denied amount.';
}

function safeStr(v) {
    if (v == null) return '';
    if (typeof v === 'string') return v;
    if (typeof v === 'object') return v.action || v.text || v.description || v.recommendation || v.message || JSON.stringify(v);
    return String(v);
}

function buildRecommendations(data, p) {
    var llm    = data.llm_analysis || {};
    var denial = (data.classified_denials || [])[0] || {};
    var recs   = [];

    if (Array.isArray(llm.top_recommendations)) {
        llm.top_recommendations.forEach(function(r, i) {
            recs.push({
                level:  i === 0 ? 'critical' : i === 1 ? 'important' : 'advisory',
                title:  i === 0 ? 'Immediate Action Required' : i === 1 ? 'High Priority' : 'Advisory',
                desc:   safeStr(r),
                action: i === 0 ? 'Act immediately' : null,
            });
        });
    }

    var rawActions = denial.recommended_actions || (denial.classification && denial.classification.recommended_actions) || [];
    rawActions.slice(0, 2).forEach(function(a) {
        var text = safeStr(a);
        if (!recs.find(function(r) { return r.desc === text; })) {
            recs.push({ level:'important', title:'Required Action', desc:text, action:null });
        }
    });

    if (llm.process_improvement) {
        recs.push({ level:'advisory', title:'Process Improvement', desc:safeStr(llm.process_improvement), action:'Long-term fix' });
    }

    if (!recs.length) {
        recs.push({ level:'critical', title:'Address Denial Code '+(p.denialCode||'CO-16'),
            desc:'The claim was denied for: '+(p.denialReason||'documentation issues')+'. Gather supporting documentation and resubmit.', action:'File appeal' });
        recs.push({ level:'important', title:'Verify Patient & Provider Information',
            desc:'Confirm all patient demographics, insurance ID, provider credentials, and procedure codes match insurer records.', action:null });
    }
    return recs;
}

function resetAnalysis() {
    document.getElementById('uploadIdle').style.display = 'block';
    document.getElementById('uploadProcessing').style.display = 'none';
    document.getElementById('eraResults').style.display = 'none';
    document.getElementById('agentSteps').innerHTML = '';
}

function showToast(msg, type) {
    type = type || 'info';
    var icons = { success:'fa-check-circle', error:'fa-exclamation-circle', info:'fa-info-circle' };
    var t = document.createElement('div');
    t.className = 'toast toast-' + type;
    t.innerHTML = '<i class="fas ' + (icons[type]||icons.info) + '"></i><span>' + esc(msg) + '</span>';
    document.getElementById('toastContainer').appendChild(t);
    setTimeout(function() { if (t.parentNode) t.parentNode.removeChild(t); }, 3500);
}

function initials2(name) {
    return (name||'?').split(' ').map(function(w) { return w[0]; }).join('').slice(0,2).toUpperCase();
}
function esc(s) {
    if (s == null) return '';
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
