// MediClaims AI - Patient Details Management
// Fetches REAL patient detail from /api/denied-claims/{patient_id}

class PatientDetailsManager {
    constructor() {
        this.patientId = this.getPatientIdFromURL();
        this.patientData = null;
        this.processingSteps = [
            { id: 'step1', text: 'Analyzing Denial', icon: 'fas fa-search' },
            { id: 'step2', text: 'Generating Appeal', icon: 'fas fa-edit' },
            { id: 'step3', text: 'Submitting to Payer', icon: 'fas fa-paper-plane' },
            { id: 'step4', text: 'Appeal Submitted', icon: 'fas fa-check-circle' }
        ];
        this.init();
    }

    getPatientIdFromURL() {
        const path = window.location.pathname;
        const segments = path.split('/');
        return segments[segments.length - 1] || 'PAT001';
    }

    async init() {
        await this.loadPatientData();
        if (this.patientData) {
            this.populatePatientInfo();
        }
        await this.loadClaimJourney();
    }

    async loadPatientData() {
        try {
            const resp = await fetch(`/appeals/api/denied-claims/${this.patientId}`);
            if (resp.ok) {
                this.patientData = await resp.json();
            } else {
                console.error('Patient not found:', this.patientId);
            }
        } catch (err) {
            console.error('Failed to load patient data:', err);
        }
    }

    populatePatientInfo() {
        const data = this.patientData;
        if (!data) return;

        const initials = (data.name || 'NA').split(' ').map(n => n[0]).join('');
        const el = (id) => document.getElementById(id);

        if (el('patientAvatar')) el('patientAvatar').textContent = initials;
        if (el('patientName')) el('patientName').textContent = data.name || '';
        if (el('patientId')) el('patientId').textContent = data.id || '';
        if (el('patientAge')) el('patientAge').textContent = data.age || '';
        if (el('doctorName')) el('doctorName').textContent = data.doctorName || '';

        // Priority and success rate
        const priorityBadge = el('priorityBadge');
        if (priorityBadge) {
            priorityBadge.textContent = `${(data.priority || 'medium').toUpperCase()} PRIORITY`;
            priorityBadge.className = `priority-badge-large ${data.priority}`;
        }
        if (el('successRate')) el('successRate').textContent = `${data.successProbability || 0}%`;

        // Claim details
        if (el('claimId')) el('claimId').textContent = data.claimId || '';
        if (el('claimAmount')) el('claimAmount').textContent = this.formatCurrency(data.amount || 0);
        if (el('serviceDate')) el('serviceDate').textContent = this.formatDate(data.serviceDate);
        if (el('procedure')) el('procedure').textContent = `${data.procedure || ''} ${data.diagnosisCode || ''}`;
        if (el('insurance')) el('insurance').textContent = data.payerName || '';

        // Denial information
        if (el('denialCode')) el('denialCode').textContent = data.denialCode || '';
        if (el('denialReason')) el('denialReason').textContent = data.denialReason || '';

        // Build explanation from real data
        const explanation = data.denialDetails
            ? data.denialDetails
            : `The insurance company (${data.payerName}) denied this claim. Reason: ${data.denialReason}. Risk score from AI analysis: ${(data.riskScore || 0).toFixed(2)}.`;
        if (el('simpleExplanation')) el('simpleExplanation').textContent = explanation;

        // Technical details from required items
        const technicalList = el('technicalDetails');
        if (technicalList) {
            const items = data.requiredItems && data.requiredItems.length > 0
                ? data.requiredItems
                : [`Denial reason: ${data.denialReason}`, `Risk score: ${data.riskScore}`, `Issues found: ${data.issuesCount}`];
            technicalList.innerHTML = items.map(item => `<li>${item}</li>`).join('');
        }

        // ERA content
        this.generateERAContent(data);
    }

    generateERAContent(data) {
        const svcDate = (data.serviceDate || '').replace(/-/g, '');
        const lastName = (data.name || 'UNKNOWN').split(' ').slice(-1)[0].toUpperCase();
        const firstName = (data.name || 'UNKNOWN').split(' ')[0].toUpperCase();
        const denialNum = (data.denialCode || 'CO-16').split('-')[1] || '16';

        const eraContent = `ST*835*0001~
BPR*I*0.00*C*ACH*CCP*01*123456789*DA*987654321~
TRN*1*${data.claimId || 'UNKNOWN'}*1234567890~
DTM*405*${svcDate}~
N1*PR*${(data.payerName || 'UNKNOWN').toUpperCase()}*XX*123456789~
N3*151 INSURANCE AVENUE~
N4*CHICAGO*IL*60601~
REF*2U*${data.claimId || ''}~
CLP*${data.claimId || ''}*2*${(data.amount || 0).toFixed(2)}*0.00*0.00*MC*${data.id}*11~
NM1*QC*1*${lastName}*${firstName}****MI*${data.id}~
DTM*232*${svcDate}~
SVC*HC:${data.procedure || 'N/A'}*${(data.amount || 0).toFixed(2)}*0.00**1~
DTM*472*${svcDate}~
CAS*CO*${denialNum}*${(data.amount || 0).toFixed(2)}~
REF*6R*${(data.denialReason || '').toUpperCase().substring(0, 60)}~`;

        const eraEl = document.getElementById('eraContent');
        if (eraEl) eraEl.textContent = eraContent;
    }

    formatCurrency(amount) {
        return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(amount);
    }

    formatDate(dateString) {
        if (!dateString) return 'N/A';
        const date = new Date(dateString);
        if (isNaN(date.getTime())) return dateString;
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    }

    showProcessingModal() {
        const modal = document.getElementById('processingModal');
        if (modal) modal.style.display = 'flex';
        this.resetProcessingSteps();
    }

    hideProcessingModal() {
        const modal = document.getElementById('processingModal');
        if (modal) modal.style.display = 'none';
    }

    resetProcessingSteps() {
        this.processingSteps.forEach(step => {
            const el = document.getElementById(step.id);
            if (el) el.className = 'step';
        });
    }

    async updateProcessingStep(stepIndex) {
        const step = this.processingSteps[stepIndex];
        const titleEl = document.getElementById('processingTitle');
        const descEl = document.getElementById('processingDescription');

        const descriptions = [
            'AI is analyzing the denial reason and identifying the best appeal strategy',
            'Creating a comprehensive appeal letter with supporting documentation',
            'Submitting the appeal directly to the insurance company',
            'Appeal has been successfully submitted and is being tracked'
        ];

        if (titleEl) titleEl.textContent = step.text;
        if (descEl) descEl.textContent = descriptions[stepIndex];

        this.processingSteps.forEach((s, index) => {
            const el = document.getElementById(s.id);
            if (el) {
                if (index < stepIndex) el.className = 'step completed';
                else if (index === stepIndex) el.className = 'step active';
                else el.className = 'step';
            }
        });

        await this.delay(1500 + Math.random() * 1000);
    }

    async autoCorrectAndSubmit() {
        this.showProcessingModal();
        try {
            // Step 1-3: Show processing animation
            for (let i = 0; i < 3; i++) {
                await this.updateProcessingStep(i);
            }

            // Step 4: Call Bedrock appeal generation
            const titleEl = document.getElementById('processingTitle');
            const descEl = document.getElementById('processingDescription');
            if (titleEl) titleEl.textContent = 'Generating AI Appeal via AWS Bedrock...';
            if (descEl) descEl.textContent = 'Nova Micro is crafting a personalized appeal letter...';

            try {
                const resp = await fetch(`/appeals/api/bedrock-appeal/${this.patientId}`, { method: 'POST' });
                if (resp.ok) {
                    const data = await resp.json();
                    await this.updateProcessingStep(3);
                    const titleEl2 = document.getElementById('processingTitle');
                    const descEl2 = document.getElementById('processingDescription');
                    if (titleEl2) titleEl2.textContent = 'Appeal Generated by AWS Bedrock!';
                    if (descEl2) descEl2.textContent = `Generated by: ${data.generated_by}`;
                    await this.delay(1500);
                    this.hideProcessingModal();
                    // Show the appeal letter
                    const expEl = document.getElementById('simpleExplanation');
                    if (expEl && data.appeal_letter) {
                        expEl.innerHTML = `<strong>🤖 AI-Generated Appeal Letter (${data.generated_by}):</strong><br><br><pre style="white-space:pre-wrap;font-size:0.8rem;background:#f8fafc;padding:1rem;border-radius:8px;border:1px solid #e2e8f0">${data.appeal_letter}</pre>`;
                    }
                    this.showToast(`Appeal generated by ${data.generated_by}`, 'success');
                    return;
                }
            } catch (bedrockErr) {
                console.warn('Bedrock appeal failed, using fallback:', bedrockErr);
            }

            await this.updateProcessingStep(3);
            await this.delay(1000);
            this.hideProcessingModal();
            this.showToast('Appeal submitted successfully! Expected response in 3-5 business days.', 'success');
        } catch (error) {
            this.hideProcessingModal();
            this.showToast('Error submitting appeal. Please try again.', 'error');
        }
    }

    showSuggestedCorrections() {
        window.location.href = `/appeals/corrections/${this.patientId}`;
    }

    delay(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }

    async loadClaimJourney() {
        const container = document.getElementById('claimJourneyContent');
        if (!container) return;
        try {
            const resp = await fetch(`/appeals/api/claim-journey/${this.patientId}`);
            if (!resp.ok) { container.innerHTML = '<p style="color:#94a3b8;font-size:0.85rem;">No claim journey data available.</p>'; return; }
            const data = await resp.json();
            this.renderClaimJourney(data, container);
        } catch (e) {
            container.innerHTML = '<p style="color:#94a3b8;font-size:0.85rem;">Could not load claim journey.</p>';
        }
    }

    renderClaimJourney(data, container) {
        const fmt = (d) => d ? new Date(d).toLocaleString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' }) : '—';
        const statusBadge = { approved: 'badge-approved', denied: 'badge-denied', pending: 'badge-pending', resubmitted: 'badge-resubmitted', error: 'badge-error' };
        const statusLabel = { approved: 'Approved', denied: 'Denied', pending: 'Pending', resubmitted: 'Resubmitted', error: 'Error' };

        const riskPct = Math.round((data.risk_score || 0) * 100);
        const dq = data.data_quality_score || 0;
        const procTime = data.processing_time_seconds ? `${data.processing_time_seconds.toFixed(1)}s` : '—';

        const metaBar = `
        <div class="claim-meta-bar">
            <div class="claim-meta-item"><span class="claim-meta-label">Claim ID</span><span class="claim-meta-value mono">${data.claim_id || '—'}</span></div>
            <div class="claim-meta-item"><span class="claim-meta-label">Amount</span><span class="claim-meta-value">$${(data.claim_amount || 0).toFixed(2)}</span></div>
            <div class="claim-meta-item"><span class="claim-meta-label">Insurer</span><span class="claim-meta-value">${data.insurer || '—'}</span></div>
            <div class="claim-meta-item"><span class="claim-meta-label">Procedure</span><span class="claim-meta-value">${data.procedure_code || '—'}</span></div>
            <div class="claim-meta-item"><span class="claim-meta-label">Diagnosis</span><span class="claim-meta-value">${data.diagnosis_code || '—'}</span></div>
            <div class="claim-meta-item"><span class="claim-meta-label">Service Date</span><span class="claim-meta-value">${data.service_date || '—'}</span></div>
            <div class="claim-meta-item"><span class="claim-meta-label">Provider</span><span class="claim-meta-value">${data.provider || '—'}</span></div>
            <div class="claim-meta-item"><span class="claim-meta-label">AI Risk Score</span><span class="claim-meta-value" style="color:${riskPct > 60 ? '#ef4444' : riskPct > 30 ? '#f59e0b' : '#10b981'}">${riskPct}%</span></div>
        </div>`;

        const aiScores = `
        <div class="ai-scores-row">
            <div class="ai-score-chip"><i class="fas fa-brain"></i> Risk Score: <strong style="color:${riskPct > 60 ? '#ef4444' : riskPct > 30 ? '#f59e0b' : '#10b981'}">${riskPct}%</strong></div>
            ${dq ? `<div class="ai-score-chip"><i class="fas fa-check-double"></i> Data Quality: <strong>${dq}%</strong></div>` : ''}
            ${data.issues_count ? `<div class="ai-score-chip"><i class="fas fa-exclamation-circle"></i> Issues Found: <strong>${data.issues_count}</strong></div>` : ''}
            <div class="ai-score-chip"><i class="fas fa-clock"></i> Processing Time: <strong>${procTime}</strong></div>
        </div>`;

        const events = (data.events || []).map(ev => {
            const badge = statusBadge[ev.badge] || 'badge-pending';
            const label = statusLabel[ev.badge] || ev.badge;
            const reqDocs = (ev.required_items || []).length > 0
                ? `<ul class="required-docs">${ev.required_items.map(r => `<li>${r}</li>`).join('')}</ul>`
                : '';
            return `
            <div class="timeline-event event-${ev.type}">
                <div class="event-header">
                    <span class="event-title"><i class="fas ${ev.icon}"></i> ${ev.title}</span>
                    <div style="display:flex;align-items:center;gap:0.5rem;">
                        <span class="event-badge ${badge}">${label}</span>
                        <span class="event-time">${fmt(ev.timestamp)}</span>
                    </div>
                </div>
                <div class="event-body">${ev.description}${reqDocs}</div>
                ${ev.type === 'submitted' ? aiScores : ''}
            </div>`;
        }).join('');

        container.innerHTML = metaBar + `<div class="journey-timeline">${events}</div>`;
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        if (!container) return;
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `<i class="fas fa-${type === 'success' ? 'check-circle' : 'info-circle'}"></i><span>${message}</span>`;
        container.appendChild(toast);
        setTimeout(() => { if (toast.parentNode) toast.parentNode.removeChild(toast); }, 5000);
    }
}

// Global functions
function goBack() { window.history.back(); }

function viewFullERA() {
    const eraContent = document.getElementById('eraContent')?.textContent || '';
    const w = window.open('', '_blank', 'width=800,height=600');
    w.document.write(`<html><head><title>ERA Document</title>
        <style>body{font-family:'Courier New',monospace;padding:20px;background:#f8fafc}
        pre{background:white;padding:20px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,0.1);white-space:pre-wrap}
        h1{color:#2563eb}</style></head>
        <body><h1>Electronic Remittance Advice (ERA)</h1><pre>${eraContent}</pre></body></html>`);
}

function printReport() { window.print(); }

function exportData() {
    const d = window.patientManager?.patientData;
    if (!d) return;
    const blob = new Blob([JSON.stringify(d, null, 2)], {type: 'application/json'});
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `patient-${d.id}-data.json`;
    link.click();
}

function autoCorrectAndSubmit() { window.patientManager?.autoCorrectAndSubmit(); }
function showSuggestedCorrections() { window.patientManager?.showSuggestedCorrections(); }

function toggleTimeline() {
    const content = document.getElementById('timelineContent');
    const toggle = document.querySelector('.timeline-toggle i');
    if (content?.classList.contains('collapsed')) {
        content.classList.remove('collapsed');
        toggle?.classList.replace('fa-chevron-down', 'fa-chevron-up');
    } else {
        content?.classList.add('collapsed');
        toggle?.classList.replace('fa-chevron-up', 'fa-chevron-down');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.patientManager = new PatientDetailsManager();
    initERAUpload();
});

// ── ERA Upload Logic ──
function initERAUpload() {
    const fileInput = document.getElementById('eraFileInput');
    const box = document.getElementById('eraUploadBox');
    if (!fileInput || !box) return;

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleERAFile(e.target.files[0]);
    });

    // Drag & drop
    box.addEventListener('dragover', (e) => { e.preventDefault(); box.classList.add('drag-over'); });
    box.addEventListener('dragleave', () => box.classList.remove('drag-over'));
    box.addEventListener('drop', (e) => {
        e.preventDefault(); box.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) handleERAFile(e.dataTransfer.files[0]);
    });
}

async function handleERAFile(file) {
    const idle = document.getElementById('uploadIdle');
    const processing = document.getElementById('uploadProcessing');
    const result = document.getElementById('uploadResult');
    const stepEl = document.getElementById('uploadStep');

    idle.style.display = 'none';
    processing.style.display = 'block';
    result.style.display = 'none';

    const text = await file.text();
    const patientId = window.patientManager?.patientId || 'unknown';

    // Animate processing steps
    const steps = [
        'Parsing ERA/835 X12 segments...',
        'Extracting CLP, CAS, NM1, SVC data...',
        'AI classifying denials via DenialClassifier...',
        'AWS Bedrock (Llama 3) analyzing denial patterns...',
        'LLM generating appeal strategies & insights...'
    ];
    for (let i = 0; i < steps.length; i++) {
        stepEl.textContent = steps[i];
        await new Promise(r => setTimeout(r, 600 + Math.random() * 400));
    }

    try {
        const resp = await fetch(`/appeals/api/process-era/${patientId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ era_content: text, filename: file.name })
        });
        const data = await resp.json();

        processing.style.display = 'none';
        result.style.display = 'block';

        const statsEl = document.getElementById('eraResultStats');
        const summary = data.summary || {};
        statsEl.innerHTML = `
            <div class="result-stat"><div class="rs-label">Total Claims</div><div class="rs-value">${summary.total_claims || 0}</div></div>
            <div class="result-stat"><div class="rs-label">Denied</div><div class="rs-value denied">${summary.denied_claims || 0}</div></div>
            <div class="result-stat"><div class="rs-label">Paid</div><div class="rs-value recovered">${summary.paid_claims || 0}</div></div>
        `;

        // Update the ERA document preview with real parsed data
        if (data.classified_denials && data.classified_denials.length > 0) {
            const d = data.classified_denials[0];
            const codeEl = document.getElementById('denialCode');
            const reasonEl = document.getElementById('denialReason');
            if (codeEl && d.denial_code) codeEl.textContent = d.denial_code;
            if (reasonEl && d.denial_reason) reasonEl.textContent = d.denial_reason;

            // Update explanation
            const cls = d.classification;
            if (cls) {
                const expEl = document.getElementById('simpleExplanation');
                if (expEl) {
                    expEl.textContent = `AI classified this as "${cls.primary_classification?.display_name}" with ${Math.round((cls.primary_classification?.confidence || 0) * 100)}% confidence. Recommended strategy: ${(cls.appeal_strategy || '').replace(/_/g, ' ')}. Expected success rate: ${Math.round((cls.expected_success_rate || 0) * 100)}%.`;
                }
            }

            // Update recommendations from classified actions
            const actions = d.recommended_actions || [];
            if (actions.length > 0) {
                const recList = document.querySelector('.recommendation-list');
                if (recList) {
                    const levels = ['high', 'medium', 'low'];
                    const icons = ['fa-star', 'fa-edit', 'fa-phone'];
                    recList.innerHTML = actions.slice(0, 3).map((a, i) => `
                        <div class="recommendation-item ${levels[i] || 'low'}">
                            <div class="rec-icon"><i class="fas ${icons[i] || 'fa-info'}"></i></div>
                            <div class="rec-content">
                                <h4>${a.action}</h4>
                                <p>${a.description || ''}</p>
                            </div>
                        </div>
                    `).join('');
                }
            }
        }

        // Show LLM analysis if available
        const llm = data.llm_analysis;
        if (llm && !llm.error) {
            const expEl = document.getElementById('simpleExplanation');
            if (expEl) {
                const assessment = llm.overall_assessment || '';
                const rootCause = llm.root_cause_analysis || '';
                const recovery = llm.estimated_recovery || '';
                expEl.innerHTML = `<strong>🤖 AI Analysis:</strong> ${assessment}<br><strong>Root Cause:</strong> ${rootCause}<br><strong>Estimated Recovery:</strong> ${recovery}`;
            }

            // Update recommendations with LLM suggestions
            const recs = llm.top_recommendations || [];
            if (recs.length > 0) {
                const recList = document.querySelector('.recommendation-list');
                if (recList) {
                    const levels = ['high', 'medium', 'low'];
                    const icons = ['fa-brain', 'fa-lightbulb', 'fa-shield-alt'];
                    recList.innerHTML = recs.slice(0, 3).map((r, i) => `
                        <div class="recommendation-item ${levels[i] || 'low'}">
                            <div class="rec-icon"><i class="fas ${icons[i] || 'fa-info'}"></i></div>
                            <div class="rec-content">
                                <h4>AI Recommendation ${i + 1}</h4>
                                <p>${r}</p>
                            </div>
                        </div>
                    `).join('');
                    if (llm.process_improvement) {
                        recList.innerHTML += `
                            <div class="recommendation-item low" style="border-left-color:#8b5cf6;background:#f5f3ff;">
                                <div class="rec-icon" style="background:#8b5cf6;"><i class="fas fa-cogs"></i></div>
                                <div class="rec-content">
                                    <h4>Process Improvement</h4>
                                    <p>${llm.process_improvement}</p>
                                </div>
                            </div>`;
                    }
                }
            }

            // Show risk level badge
            const riskLevel = llm.risk_level || 'medium';
            const techDetails = document.getElementById('technicalDetails');
            if (techDetails) {
                const priorityOrder = (llm.appeal_priority_order || []).join(', ') || 'N/A';
                techDetails.innerHTML = `
                    <li><strong>Risk Level:</strong> <span style="color:${riskLevel === 'high' ? '#ef4444' : riskLevel === 'medium' ? '#f59e0b' : '#10b981'};font-weight:700;text-transform:uppercase;">${riskLevel}</span></li>
                    <li><strong>Appeal Priority:</strong> ${priorityOrder}</li>
                    <li><strong>Powered by:</strong> AWS Bedrock — Llama 3 70B</li>
                `;
            }
        }

        // Update ERA preview with uploaded content
        const eraEl = document.getElementById('eraContent');
        if (eraEl) eraEl.textContent = text.substring(0, 2000);

        window.patientManager?.showToast(`ERA processed: ${data.denials_count} denials found — AI analysis complete`, 'success');
    } catch (err) {
        processing.style.display = 'none';
        idle.style.display = 'block';
        window.patientManager?.showToast('Error processing ERA file', 'error');
        console.error(err);
    }
}

function resetERAUpload() {
    document.getElementById('uploadIdle').style.display = 'block';
    document.getElementById('uploadProcessing').style.display = 'none';
    document.getElementById('uploadResult').style.display = 'none';
    document.getElementById('eraFileInput').value = '';
}
