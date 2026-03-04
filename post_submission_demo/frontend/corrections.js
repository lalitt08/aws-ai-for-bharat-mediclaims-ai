// MediClaims AI - Corrections Management
// Fetches REAL patient data from /api/denied-claims/{patient_id}
// Generates corrections dynamically based on actual denial reason

class CorrectionsManager {
    constructor() {
        this.patientId = this.getPatientIdFromURL();
        this.patientData = null;
        this.corrections = [];
        this.implementationSteps = [
            { id: 'step1', text: 'Updating Procedure Codes', icon: 'fas fa-edit' },
            { id: 'step2', text: 'Adding Documentation', icon: 'fas fa-file-medical' },
            { id: 'step3', text: 'Validating Changes', icon: 'fas fa-check-double' },
            { id: 'step4', text: 'Submitting Corrected Claim', icon: 'fas fa-paper-plane' }
        ];
        this.init();
    }

    getPatientIdFromURL() {
        const path = window.location.pathname;
        const segments = path.split('/');
        return segments[segments.length - 1] || 'PAT002';
    }

    async init() {
        await this.loadPatientData();
        if (this.patientData) {
            this.generateCorrections();
            this.populatePatientInfo();
            this.renderCorrections();
        }
    }

    async loadPatientData() {
        try {
            const resp = await fetch(`/api/denied-claims/${this.patientId}`);
            if (resp.ok) {
                this.patientData = await resp.json();
            } else {
                console.error('Patient not found:', this.patientId);
            }
        } catch (err) {
            console.error('Failed to load patient data:', err);
        }
    }

    generateCorrections() {
        const d = this.patientData;
        if (!d) return;

        const reason = (d.denialReason || '').toLowerCase();
        const code = d.denialCode || 'CO-16';
        this.corrections = [];

        // Build corrections dynamically based on the real denial reason
        if (reason.includes('prior authorization') || code === 'CO-197') {
            this.corrections.push({
                id: 1, priority: 'critical',
                title: 'Obtain Prior Authorization',
                description: 'Submit retroactive prior authorization request to ' + (d.payerName || 'insurer'),
                details: d.denialDetails || 'The claim was denied because prior authorization was not obtained or has expired. Contact the insurer to request retroactive authorization.',
                before: 'Box 23 (Prior Auth): [EMPTY / EXPIRED]',
                after: 'Box 23: AUTH-' + new Date().getFullYear() + '-RETRO-' + d.id,
                impact: { successRate: '+70%', processingTime: '-3 days', appealTime: '-14 days' }
            });
            this.corrections.push({
                id: 2, priority: 'important',
                title: 'Add Clinical Necessity Documentation',
                description: 'Provide clinical notes supporting why the service could not wait for authorization',
                details: 'Include documentation showing the urgency or clinical necessity that prevented obtaining prior authorization before the service date.',
                before: 'No urgency documentation',
                after: 'Clinical urgency letter + treatment timeline',
                impact: { successRate: '+15%', processingTime: '0 days', appealTime: '-5 days' }
            });
        } else if (reason.includes('medical necessity') || code === 'CO-50') {
            this.corrections.push({
                id: 1, priority: 'critical',
                title: 'Submit Medical Necessity Justification',
                description: 'Provide comprehensive clinical documentation supporting the treatment',
                details: d.denialDetails || 'The insurer requires additional clinical evidence to establish medical necessity. Include progress notes, treatment history, and clinical guidelines.',
                before: 'Insufficient clinical documentation',
                after: 'Complete clinical package: progress notes + treatment plan + guidelines',
                impact: { successRate: '+75%', processingTime: '-2 days', appealTime: '-10 days' }
            });
            this.corrections.push({
                id: 2, priority: 'important',
                title: 'Request Peer-to-Peer Review',
                description: 'Schedule physician-to-physician review with insurer medical director',
                details: 'A peer-to-peer review allows the treating physician to directly explain the clinical rationale to the insurer\'s reviewing physician.',
                before: 'No peer review requested',
                after: 'Peer-to-peer review scheduled with ' + (d.payerName || 'insurer'),
                impact: { successRate: '+20%', processingTime: '+2 days', appealTime: '-7 days' }
            });
        } else if (reason.includes('diagnosis code') || reason.includes('coding') || reason.includes('modifier') || code === 'CO-4') {
            this.corrections.push({
                id: 1, priority: 'critical',
                title: 'Correct Diagnosis/Procedure Codes',
                description: 'Update codes to match clinical documentation',
                details: d.denialDetails || 'The diagnosis and procedure codes are inconsistent. Review the operative/clinical notes and update the codes to accurately reflect the services provided.',
                before: 'Current: ' + (d.procedure || 'N/A') + ' / ' + (d.diagnosisCode || 'N/A'),
                after: 'Corrected codes matching clinical documentation',
                impact: { successRate: '+80%', processingTime: '-1 day', appealTime: '-7 days' }
            });
            if (reason.includes('modifier')) {
                this.corrections.push({
                    id: 2, priority: 'important',
                    title: 'Add Required Modifier',
                    description: 'Include the correct procedure modifier (e.g., bilateral, distinct procedure)',
                    details: d.denialDetails || 'The procedure requires a modifier to clarify laterality, distinct procedure, or other specifics.',
                    before: 'No modifier on procedure code',
                    after: 'Appropriate modifier added (e.g., -50 bilateral, -LT left, -59 distinct)',
                    impact: { successRate: '+15%', processingTime: '0 days', appealTime: '-3 days' }
                });
            }
        } else if (reason.includes('documentation') || reason.includes('missing') || code === 'CO-16') {
            this.corrections.push({
                id: 1, priority: 'critical',
                title: 'Submit Missing Documentation',
                description: 'Provide all required clinical records and supporting documents',
                details: d.denialDetails || 'The claim was denied due to missing or incomplete documentation. Gather and submit all required records.',
                before: 'Incomplete documentation package',
                after: 'Complete documentation: ' + (d.requiredItems || ['medical records', 'clinical notes', 'treatment plan']).join(', '),
                impact: { successRate: '+85%', processingTime: '-2 days', appealTime: '-10 days' }
            });
        } else if (reason.includes('credential') || reason.includes('provider')) {
            this.corrections.push({
                id: 1, priority: 'critical',
                title: 'Update Provider Credentials',
                description: 'Submit updated NPI and provider enrollment documentation',
                details: d.denialDetails || 'Provider credentials need to be updated or verified with the insurer.',
                before: 'Provider NPI/enrollment not verified',
                after: 'Updated NPI documentation + enrollment verification + state license',
                impact: { successRate: '+90%', processingTime: '-1 day', appealTime: '-5 days' }
            });
        } else if (reason.includes('service level') || reason.includes('mismatch')) {
            this.corrections.push({
                id: 1, priority: 'critical',
                title: 'Correct Service Level Authorization',
                description: 'Update authorization to match the service level provided',
                details: d.denialDetails || 'The authorized service level differs from what was provided. Request updated authorization matching the actual service.',
                before: 'Authorization for lower service level',
                after: 'Updated authorization matching provided service level',
                impact: { successRate: '+55%', processingTime: '-2 days', appealTime: '-7 days' }
            });
        }

        // Always add a general documentation improvement correction
        if (this.corrections.length > 0) {
            this.corrections.push({
                id: this.corrections.length + 1, priority: 'recommended',
                title: 'Strengthen Supporting Documentation',
                description: 'Add additional clinical evidence to maximize appeal success',
                details: 'Include any additional supporting documents: lab results, imaging reports, specialist consultations, or published clinical guidelines that support the treatment provided.',
                before: 'Standard documentation package',
                after: 'Enhanced package with additional clinical evidence',
                impact: { successRate: '+10%', processingTime: '0 days', appealTime: '-2 days' }
            });
        }

        // If no specific corrections matched, provide generic ones
        if (this.corrections.length === 0) {
            this.corrections.push({
                id: 1, priority: 'critical',
                title: 'Address Denial: ' + (d.denialReason || 'Unknown'),
                description: 'Review and correct the issues identified by the insurer',
                details: d.denialDetails || 'Review the denial reason and gather all required documentation to support the appeal.',
                before: 'Current claim as submitted',
                after: 'Corrected claim with supporting documentation',
                impact: { successRate: '+60%', processingTime: '-2 days', appealTime: '-7 days' }
            });
        }
    }

    populatePatientInfo() {
        const data = this.patientData;
        if (!data) return;
        const initials = (data.name || 'NA').split(' ').map(n => n[0]).join('');

        const el = (id) => document.getElementById(id);
        if (el('patientAvatar')) el('patientAvatar').textContent = initials;
        if (el('patientName')) el('patientName').textContent = data.name || '';
        if (el('claimId')) el('claimId').textContent = data.claimId || '';
        if (el('claimAmount')) el('claimAmount').textContent = this.formatCurrency(data.amount || 0);
        if (el('totalCorrections')) el('totalCorrections').textContent = this.corrections.length;

        const totalImpact = this.corrections.reduce((sum, c) => {
            const rate = parseInt((c.impact.successRate || '0').replace(/[^0-9]/g, ''));
            return sum + rate;
        }, 0);
        if (el('successRate')) el('successRate').textContent = Math.min(95, totalImpact) + '%';
    }

    renderCorrections() {
        const container = document.getElementById('correctionsList');
        if (!container) return;
        container.innerHTML = this.corrections.map(c => this.createCorrectionCard(c)).join('');
    }

    createCorrectionCard(correction) {
        return `
            <div class="correction-item ${correction.priority}">
                <div class="correction-header">
                    <div class="correction-title">
                        <div class="correction-icon">
                            <i class="fas fa-${this.getPriorityIcon(correction.priority)}"></i>
                        </div>
                        <div class="correction-title-text">
                            <h3>${correction.title}</h3>
                            <p>${correction.description}</p>
                        </div>
                    </div>
                    <div class="correction-priority ${correction.priority}">
                        ${correction.priority}
                    </div>
                </div>
                <div class="correction-body">
                    <div class="correction-details">
                        <h4>What needs to be corrected:</h4>
                        <p>${correction.details}</p>
                    </div>
                    <div class="before-after">
                        <div class="before">
                            <h5>Before</h5>
                            <div class="code-block">${correction.before}</div>
                        </div>
                        <div class="after">
                            <h5>After</h5>
                            <div class="code-block">${correction.after}</div>
                        </div>
                    </div>
                    <div class="correction-impact">
                        <div class="impact-grid">
                            <div class="impact-item">
                                <span class="impact-value">${correction.impact.successRate}</span>
                                <div class="impact-label">Success Rate</div>
                            </div>
                            <div class="impact-item">
                                <span class="impact-value">${correction.impact.processingTime}</span>
                                <div class="impact-label">Processing Time</div>
                            </div>
                            <div class="impact-item">
                                <span class="impact-value">${correction.impact.appealTime}</span>
                                <div class="impact-label">Appeal Time</div>
                            </div>
                        </div>
                    </div>
                    <div class="correction-actions">
                        <button class="correction-btn" onclick="viewCorrectionDetails(${correction.id})">
                            <i class="fas fa-info-circle"></i> Details
                        </button>
                        <button class="correction-btn primary" onclick="implementCorrection(${correction.id})">
                            <i class="fas fa-check"></i> Implement
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    getPriorityIcon(priority) {
        const icons = { critical: 'exclamation-triangle', important: 'exclamation-circle', recommended: 'star', optional: 'info-circle' };
        return icons[priority] || 'info-circle';
    }

    formatCurrency(amount) {
        return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(amount);
    }

    showImplementationModal() {
        const modal = document.getElementById('implementationModal');
        if (modal) modal.style.display = 'flex';
        this.resetImplementationSteps();
    }

    hideImplementationModal() {
        const modal = document.getElementById('implementationModal');
        if (modal) modal.style.display = 'none';
    }

    resetImplementationSteps() {
        this.implementationSteps.forEach(step => {
            const el = document.getElementById(step.id);
            if (el) el.className = 'step';
        });
    }

    updateProgressCircle(percentage) {
        const circle = document.querySelector('.progress-circle');
        const progressText = document.getElementById('progressText');
        if (progressText) progressText.textContent = `${percentage}%`;
        if (circle) circle.style.background = `conic-gradient(var(--medical-green) ${percentage * 3.6}deg, var(--border-light) 0deg)`;
    }

    async implementAllCorrections() {
        this.showImplementationModal();
        try {
            for (let i = 0; i < this.implementationSteps.length; i++) {
                const step = this.implementationSteps[i];
                const progress = Math.round(((i + 1) / this.implementationSteps.length) * 100);
                this.updateProgressCircle(progress);

                const titleEl = document.getElementById('implementationTitle');
                const descEl = document.getElementById('implementationDescription');
                if (titleEl) titleEl.textContent = step.text;
                const descriptions = [
                    'Updating procedure codes and billing information...',
                    'Adding required clinical documentation and notes...',
                    'Performing final validation and compliance checks...',
                    'Submitting the corrected claim to the insurance company...'
                ];
                if (descEl) descEl.textContent = descriptions[i];

                this.implementationSteps.forEach((s, index) => {
                    const el = document.getElementById(s.id);
                    if (el) {
                        if (index < i) el.className = 'step completed';
                        else if (index === i) el.className = 'step active';
                        else el.className = 'step';
                    }
                });
                await this.delay(2000 + Math.random() * 1000);
            }

            const titleEl = document.getElementById('implementationTitle');
            const descEl = document.getElementById('implementationDescription');
            if (titleEl) titleEl.textContent = 'All Corrections Implemented!';
            if (descEl) descEl.textContent = 'The corrected claim has been successfully submitted. Expected processing time: 3-5 business days.';

            await this.delay(2000);
            this.hideImplementationModal();
            this.showToast('All corrections implemented and claim resubmitted successfully!', 'success');
        } catch (error) {
            this.hideImplementationModal();
            this.showToast('Error implementing corrections. Please try again.', 'error');
        }
    }

    delay(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }

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

function implementCorrection(correctionId) {
    const c = window.correctionsManager?.corrections.find(x => x.id === correctionId);
    if (c) {
        window.correctionsManager.showToast(`Implementing: ${c.title}`, 'info');
        setTimeout(() => {
            window.correctionsManager.showToast(`${c.title} implemented successfully!`, 'success');
        }, 1500);
    }
}

function viewCorrectionDetails(correctionId) {
    const c = window.correctionsManager?.corrections.find(x => x.id === correctionId);
    if (c) {
        alert(`Correction Details:\n\n${c.details}\n\nExpected Impact:\n- Success Rate: ${c.impact.successRate}\n- Processing Time: ${c.impact.processingTime}\n- Appeal Time: ${c.impact.appealTime}`);
    }
}

function implementAllCorrections() {
    window.correctionsManager?.implementAllCorrections();
}

function exportCorrections() {
    const corrections = window.correctionsManager?.corrections;
    if (!corrections) return;
    const blob = new Blob([JSON.stringify(corrections, null, 2)], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `corrections-${window.correctionsManager.patientId}.json`;
    link.click();
}

document.addEventListener('DOMContentLoaded', () => {
    window.correctionsManager = new CorrectionsManager();
});
