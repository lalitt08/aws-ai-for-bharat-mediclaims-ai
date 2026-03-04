// MediClaims AI - Patients Management System
// Fetches REAL denied claims from the pre-submission pipeline via /api/denied-claims

function openAnalytics() {
    window.location.href = '/analytics';
}

class PatientsManager {
    constructor() {
        this.patients = [];
        this.filteredPatients = [];
        this.init();
    }

    async init() {
        this.setupEventListeners();
        this.showLoading();
        await this.loadPatients();
        this.hideLoading();
    }

    setupEventListeners() {
        document.getElementById('priorityFilter')?.addEventListener('change', () => this.filterPatients());
        document.getElementById('payerFilter')?.addEventListener('change', () => this.filterPatients());
        document.getElementById('searchFilter')?.addEventListener('input', () => this.filterPatients());
    }

    showLoading() {
        document.getElementById('loadingOverlay').style.display = 'flex';
    }

    hideLoading() {
        document.getElementById('loadingOverlay').style.display = 'none';
    }

    async loadPatients() {
        try {
            const resp = await fetch('/api/denied-claims');
            const data = await resp.json();
            this.patients = data.denied_claims || [];
        } catch (err) {
            console.error('Failed to load denied claims:', err);
            this.patients = [];
        }

        if (this.patients.length === 0) {
            // Show a helpful empty state instead of fake data
            this.filteredPatients = [];
            this.renderPatients();
            this.updateStats();
            return;
        }

        this.filteredPatients = [...this.patients];
        this.renderPatients();
        this.updateStats();
    }

    renderPatients() {
        const grid = document.getElementById('patientsGrid');
        const emptyState = document.getElementById('emptyState');

        if (this.filteredPatients.length === 0) {
            grid.style.display = 'none';
            emptyState.style.display = 'block';
            return;
        }

        grid.style.display = 'grid';
        emptyState.style.display = 'none';
        grid.innerHTML = this.filteredPatients.map(p => this.createPatientCard(p)).join('');
    }

    createPatientCard(patient) {
        const initials = (patient.name || 'NA').split(' ').map(n => n[0]).join('');
        const formattedAmount = new Intl.NumberFormat('en-US', {
            style: 'currency', currency: 'USD'
        }).format(patient.amount || 0);

        const statusLabel = this.getStatusLabel(patient.status);
        const statusClass = this.getStatusClass(patient.status);

        return `
            <div class="patient-card" onclick="viewPatientDetails('${patient.id}')">
                <div class="patient-header">
                    <div class="patient-info">
                        <div class="patient-avatar">${initials}</div>
                        <div class="patient-details">
                            <h3>${patient.name}</h3>
                            <div class="patient-id">${patient.id} &bull; ${patient.claimId}</div>
                        </div>
                        <div class="priority-badge ${patient.priority}">${patient.priority}</div>
                    </div>
                </div>

                <div class="patient-body">
                    <div class="card-fields">
                        <div class="card-field">
                            <span class="field-label">Amount</span>
                            <span class="field-value amount">${formattedAmount}</span>
                        </div>
                        <div class="card-field">
                            <span class="field-label">Service Date</span>
                            <span class="field-value">${this.formatDate(patient.serviceDate)}</span>
                        </div>
                        <div class="card-field">
                            <span class="field-label">Success Rate</span>
                            <span class="field-value">${patient.successProbability}%</span>
                        </div>
                        <div class="card-field">
                            <span class="field-label">Status</span>
                            <span class="field-value ${statusClass}">${statusLabel}</span>
                        </div>
                    </div>

                    <div class="denial-info">
                        <div class="denial-title">
                            <i class="fas fa-exclamation-triangle"></i>
                            <span>Denial: ${patient.denialCode}</span>
                        </div>
                        <div class="denial-reason">${patient.denialReason}</div>
                    </div>

                    <div class="insurance-info">
                        <div class="insurance-logo">${this.getPayerLogo(patient.payer)}</div>
                        <div>
                            <div style="font-weight: 600; font-size: 0.85rem;">${patient.payerName}</div>
                            <div style="font-size: 0.75rem; color: hsl(215 15% 45%);">${patient.procedure}</div>
                        </div>
                    </div>

                    <button class="see-details-btn" onclick="event.stopPropagation(); viewPatientDetails('${patient.id}')">
                        <i class="fas fa-search"></i>
                        See Details & ERA Analysis
                    </button>
                </div>
            </div>
        `;
    }

    getStatusLabel(status) {
        const map = {
            'appeal_resubmitted': 'Appeal Resubmitted',
            'appeal_resubmitted_low_confidence': 'Resubmitted (Low Confidence)',
            'appeal_generated': 'Appeal Generated',
            'awaiting_patient_data_update': 'Awaiting Data Update',
            'denied': 'Denied',
            'rejected': 'Rejected',
            'resubmission': 'Resubmission Pending',
        };
        return map[status] || status;
    }

    getStatusClass(status) {
        if (status && status.includes('resubmitted')) return 'status-resubmitted';
        if (status && status.includes('appeal')) return 'status-appeal';
        return 'status-denied';
    }

    getPayerLogo(payer) {
        const logos = {
            'aetna': 'AET', 'united': 'UHC', 'bluecross': 'BCBS', 'cigna': 'CIG'
        };
        return logos[(payer || '').toLowerCase()] || 'INS';
    }

    formatDate(dateString) {
        if (!dateString) return 'N/A';
        const date = new Date(dateString);
        if (isNaN(date.getTime())) return dateString;
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    }

    filterPatients() {
        const priorityFilter = document.getElementById('priorityFilter').value;
        const payerFilter = document.getElementById('payerFilter').value;
        const searchFilter = (document.getElementById('searchFilter').value || '').toLowerCase();

        this.filteredPatients = this.patients.filter(patient => {
            const matchesPriority = !priorityFilter || patient.priority === priorityFilter;
            const matchesPayer = !payerFilter || (patient.payer || '').toLowerCase().includes(payerFilter);
            const matchesSearch = !searchFilter ||
                (patient.name || '').toLowerCase().includes(searchFilter) ||
                (patient.claimId || '').toLowerCase().includes(searchFilter) ||
                (patient.id || '').toLowerCase().includes(searchFilter) ||
                (patient.denialReason || '').toLowerCase().includes(searchFilter);
            return matchesPriority && matchesPayer && matchesSearch;
        });

        this.renderPatients();
    }

    updateStats() {
        const urgentCount = this.patients.filter(p => p.priority === 'high').length;
        const totalDenied = this.patients.length;
        const totalAmount = this.patients.reduce((sum, p) => sum + (p.amount || 0), 0);
        const avgSuccess = this.patients.length > 0
            ? Math.round(this.patients.reduce((sum, p) => sum + (p.successProbability || 0), 0) / this.patients.length)
            : 0;

        const el = (id) => document.getElementById(id);
        if (el('urgentClaims')) el('urgentClaims').textContent = urgentCount;
        if (el('totalDenied')) el('totalDenied').textContent = totalDenied;
        if (el('recoveryAmount')) el('recoveryAmount').textContent = '$' + Math.round(totalAmount / 1000) + 'K';
        if (el('avgSuccess')) el('avgSuccess').textContent = avgSuccess + '%';
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        if (!container) return;
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `<i class="fas fa-${type === 'success' ? 'check-circle' : 'info-circle'}"></i><span>${message}</span>`;
        container.appendChild(toast);
        setTimeout(() => { if (toast.parentNode) toast.parentNode.removeChild(toast); }, 4000);
    }
}

// Global functions
function viewPatientDetails(patientId) {
    window.location.href = `/patient-details/${patientId}`;
}

function loadPatients() {
    if (window.patientsManager) {
        window.patientsManager.showToast('Refreshing patient data...', 'info');
        window.patientsManager.loadPatients();
    }
}

function clearFilters() {
    document.getElementById('priorityFilter').value = '';
    document.getElementById('payerFilter').value = '';
    document.getElementById('searchFilter').value = '';
    if (window.patientsManager) window.patientsManager.filterPatients();
}

document.addEventListener('DOMContentLoaded', () => {
    window.patientsManager = new PatientsManager();
});
