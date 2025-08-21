/**
 * Agentic AI Claims Dashboard (New UI wiring)
 * - Landing -> Dashboard transition
 * - Tabs: Patients, Records, Rejected
 * - New selectors and card markup (no Bootstrap)
 */

// Application state
let patients = [];
let activeClaims = [];
let processedClaims = [];
let rejectedClaims = [];
let currentStatusFilter = 'All Status';
let userFriendlyView = true; // Toggle for user-friendly vs technical view
let lastActivities = []; // Store last activities for view switching
let metrics = {
    recoveredAmount: 0,
    claimsApplied: 0,
    activeClaims: 0,
    successClaims: 0,
    successRate: 0
};

// Base URL for API endpoints (ensure proper resolution when loaded from file://)
const API_BASE_URL = 'http://localhost:5000';

// Redirect to server if page loaded directly as file://
if (window.location.protocol === 'file:') {
    console.warn('Dashboard loaded via file://, redirecting to server...');
    window.location.href = API_BASE_URL;
}

// DOM Ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('🏥 Agentic AI Claims Dashboard Loading...');

    // Landing -> Dashboard transition
    const enterBtn = document.getElementById('enter-dashboard-btn');
    const landing = document.getElementById('landing-page');
    const dashboard = document.getElementById('dashboard');
    if (enterBtn && landing && dashboard) {
        enterBtn.addEventListener('click', () => {
            landing.style.display = 'none';
            dashboard.style.display = 'block';

            setupTabs();
            setupControls();
            initializeDashboard();
        });
    } else {
        // If landing elements not found, fallback to initializing directly
        setupTabs();
        setupControls();
        initializeDashboard();
    }
});

function setupTabs() {
    const triggers = document.querySelectorAll('.tab-trigger');
    const tabMap = {
        patients: document.getElementById('patients-tab'),
        records: document.getElementById('records-tab'),
        rejected: document.getElementById('rejected-tab')
    };
    const setActive = (key) => {
        // Toggle trigger active
        triggers.forEach(btn => btn.classList.toggle('active', btn.getAttribute('data-tab') === key));
        // Toggle content active
        Object.keys(tabMap).forEach(k => {
            const el = tabMap[k];
            if (!el) return;
            if (k === key) {
                el.classList.add('active');
                el.style.display = 'block';
            } else {
                el.classList.remove('active');
                el.style.display = 'none';
            }
        });
        
        // Reference-like transitions on tab change
        animateTabChange();
        if (key === 'records') {
            setTimeout(animateMetricCards, 100);
            setTimeout(animateActivityFeed, 200);
            // Start real-time activity refresh when Records tab is active
            startActivityRefresh();
        } else {
            // Stop activity refresh when leaving Records tab to save resources
            stopActivityRefresh();
        }
    };
    // Default to patients
    setActive('patients');
    triggers.forEach(btn => btn.addEventListener('click', () => setActive(btn.getAttribute('data-tab'))));
}

function setupControls() {
    const refreshBtn = document.getElementById('refresh-btn');
    const refreshInline = document.getElementById('refresh-inline');
    const refreshActivity = document.getElementById('refresh-activity');
    const clearActivities = document.getElementById('clear-activities');
    const toggleView = document.getElementById('toggle-view');
    const searchInput = document.getElementById('search-input');
    
    if (refreshBtn) refreshBtn.addEventListener('click', refreshData);
    if (refreshInline) refreshInline.addEventListener('click', refreshData);
    if (refreshActivity) {
        refreshActivity.addEventListener('click', () => {
            fetchAgentActivity();
            // Add visual feedback
            const icon = refreshActivity.querySelector('svg');
            if (icon) {
                icon.classList.add('animate-spin');
                setTimeout(() => icon.classList.remove('animate-spin'), 1000);
            }
        });
    }
    
    // Clear all activities
    if (clearActivities) {
        clearActivities.addEventListener('click', async () => {
            try {
                const response = await fetch('/api/clear-activities', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                if (response.ok) {
                    // Clear the UI immediately
                    updateAgentActivity([]);
                    console.log('✅ Activities cleared successfully');
                    
                    // Show success feedback
                    const icon = clearActivities.querySelector('svg');
                    if (icon) {
                        icon.classList.add('animate-pulse');
                        setTimeout(() => icon.classList.remove('animate-pulse'), 1000);
                    }
                } else {
                    console.error('❌ Failed to clear activities');
                }
            } catch (error) {
                console.error('❌ Error clearing activities:', error);
            }
        });
    }
    
    // Toggle between user-friendly and technical view
    if (toggleView) {
        toggleView.addEventListener('click', () => {
            userFriendlyView = !userFriendlyView;
            updateViewModeText();
            // Re-render activities with the new view mode
            if (lastActivities.length > 0) {
                updateAgentActivity(lastActivities);
            }
        });
    }
    
    if (refreshBtn) refreshBtn.addEventListener('click', refreshData);
    if (refreshInline) refreshInline.addEventListener('click', refreshData);
    if (refreshActivity) {
        refreshActivity.addEventListener('click', () => {
            fetchAgentActivity();
            // Add visual feedback
            const icon = refreshActivity.querySelector('svg');
            if (icon) {
                icon.classList.add('animate-spin');
                setTimeout(() => icon.classList.remove('animate-spin'), 1000);
            }
        });
    }
    
    // Add spin animation to refresh icons (reference-like)
    const spinIcon = (btn) => {
        if (!btn) return;
        const icon = btn.querySelector('svg');
        if (!icon) return;
        icon.style.animation = 'spin 1s linear';
        setTimeout(() => { icon.style.animation = ''; }, 1000);
    };
    if (refreshBtn) refreshBtn.addEventListener('click', () => spinIcon(refreshBtn));
    if (refreshInline) refreshInline.addEventListener('click', () => spinIcon(refreshInline));
    if (searchInput) {
        searchInput.addEventListener('input', () => renderPatientCards(searchInput.value.trim()));
    }

    // Status filter chips (All Status, Ready, Processing, Approved, Denied)
    const statusChips = document.querySelectorAll('#patients-tab .badge-outline, #patients-tab .badge-secondary');
    statusChips.forEach(chip => {
        chip.addEventListener('click', () => {
            // Remove active-filter from all chips
            statusChips.forEach(c => c.classList.remove('active-filter'));
            // Set active on clicked
            chip.classList.add('active-filter');
            // Update filter value
            currentStatusFilter = chip.textContent.trim();
            // Re-render with filter
            renderPatientCards();
        });
    });
}

function initializeDashboard() {
    // Initialize with basic metrics
    metrics = {
        recoveredAmount: 0,
        claimsApplied: 0,
        activeClaims: 0,
        successClaims: 0,
        successRate: 0
    };
    
    updateMetrics();

    // Show loading indicator
    const container = document.querySelector('.patient-grid');
    if (container) {
        container.innerHTML = `
            <div class="card p-6 text-center">
                <div class="text-sm text-muted-foreground mb-2">Loading Patient Data from OpenEMR...</div>
                <div class="text-xs text-muted-foreground">Reading real-time data...</div>
            </div>
        `;
    }
    
    // Test API connectivity first
    console.log('🔌 Testing API connectivity...');
    fetch(`${API_BASE_URL}/api/metrics`)
        .then(response => {
            if (response.ok) {
                console.log('✅ API connectivity test passed');
                // Load patient data from CSV files
                loadPatientData();
            } else {
                throw new Error(`API test failed: ${response.status}`);
            }
        })
        .catch(error => {
            console.error('❌ API connectivity test failed:', error);
            const grid = document.querySelector('.patient-grid');
            if (grid) {
                grid.innerHTML = `
                    <div class="card p-6 text-center">
                        <div class="text-lg font-semibold mb-2">Cannot connect to API server</div>
                        <div class="text-sm text-muted-foreground mb-4">Make sure the server is running at ${API_BASE_URL}</div>
                        <button class="btn btn-sm" onclick="initializeDashboard()">Retry Connection</button>
                    </div>
                `;
            }
            return;
        });
    
    // Do not push any placeholder activity; fetch real activity only
    fetchAgentActivity();
    
    // DISABLED: Auto-refresh for CSV data (was causing repeated loading messages)
    // Only refresh data when user explicitly clicks refresh button
    // setInterval(() => {
    //     console.log('🔄 Auto-refreshing CSV data...');
    //     refreshDataFromCSV();
    // }, 30000); // Refresh every 30 seconds
    
    // Set up auto-refresh for active claims status every 5 seconds
    setInterval(checkActiveClaimsStatus, 5000); // Check every 5 seconds
}

function updateMetrics() {
    // Update Records tab metrics
    const activeEl = document.getElementById('activeClaimsCount');
    const successEl = document.getElementById('successClaimsCount');
    const amountEl = document.getElementById('recoveredAmountTotal');
    const rateEl = document.getElementById('successRate');
    if (activeEl) activeEl.textContent = metrics.activeClaims;
    if (successEl) successEl.textContent = metrics.successClaims;
    if (amountEl) amountEl.textContent = `$${metrics.recoveredAmount.toLocaleString()}`;
    if (rateEl) rateEl.textContent = `${metrics.successRate}%`;
    const activeBadge = document.getElementById('active-badge');
    if (activeBadge) activeBadge.textContent = `${metrics.activeClaims} Active`;
}

function loadPatientData() {
    console.log('📊 Loading patient data from OpenEMR database...');
    console.log('🔍 Current window location:', window.location.href);
    const patientsUrl = `${API_BASE_URL}/api/patients`;
    console.log('🔍 Attempting to fetch from:', patientsUrl);
    
    // Clear previous error state
    const grid = document.querySelector('.patient-grid');
    if (grid) {
        grid.innerHTML = `
            <div class="card p-6 text-center">
                <div class="text-sm text-muted-foreground mb-2">Loading Patient Data...</div>
                <div class="text-xs text-muted-foreground">Fetching from ${patientsUrl}</div>
            </div>
        `;
    }
    
    fetch(patientsUrl)
        .then(response => {
            console.log('📡 API response status:', response.status);
            console.log('📡 API response ok:', response.ok);
            console.log('📡 API response headers:', response.headers);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status} - ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('📊 Raw patient data received:', data);
            console.log('📊 Data type:', typeof data);
            console.log('📊 Is Array:', Array.isArray(data));
            console.log('📊 Data length:', data ? data.length : 'N/A');
            
            if (Array.isArray(data) && data.length > 0) {
                patients = data;
                console.log(`✅ Loaded ${patients.length} patients from OpenEMR database`);
                console.log('📊 First patient:', patients[0]);
                renderPatientCards();
                showSuccess(`Loaded ${patients.length} patients from OpenEMR database`);
            } else if (Array.isArray(data) && data.length === 0) {
                console.warn('⚠️ Empty patient data array received');
                const grid = document.querySelector('.patient-grid');
                if (grid) {
                    grid.innerHTML = `
                        <div class="card p-6 text-center">
                            <div class="text-sm">No patients found.</div>
                        </div>
                    `;
                }
            } else {
                console.error('❌ Invalid patient data format received:', data);
                const grid = document.querySelector('.patient-grid');
                if (grid) {
                    grid.innerHTML = `
                        <div class="card p-6 text-center">
                            <div class="text-sm">Error: Invalid patient data format. Expected array, got ${typeof data}.</div>
                        </div>
                    `;
                }
            }
        })
        .catch(error => {
            console.error('❌ Error loading patient data:', error);
            console.error('❌ Error details:', error.message);
            console.error('❌ Error stack:', error.stack);
            
            // Display detailed error information
            const grid = document.querySelector('.patient-grid');
            if (grid) {
                grid.innerHTML = `
                    <div class="card p-6 text-center">
                        <div class="text-lg font-semibold mb-2">Failed to load patient data</div>
                        <div class="text-sm text-muted-foreground mb-4"><strong>Error:</strong> ${error.message}</div>
                        <div class="text-xs text-muted-foreground mb-4"><strong>URL:</strong> ${API_BASE_URL}/api/patients</div>
                        <button class="btn btn-sm" onclick="loadPatientData()">Try Again</button>
                    </div>
                `;
            }
        });
}

function refreshDataFromCSV() {
    console.log('🔄 Refreshing data from real-time OpenEMR database...');
    
    // First reload the data on the server
    const reloadUrl = `${API_BASE_URL}/api/reload-data`;
    fetch(reloadUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(result => {
        if (result.success) {
            console.log('✅ Server data reloaded:', result.message);
            // Now fetch the updated patient data
            loadPatientData();
        } else {
            console.error('❌ Failed to reload server data:', result.error);
            showError('Failed to reload data from CSV files');
        }
    })
    .catch(error => {
        console.error('❌ Error reloading data:', error);
        // Fallback to just reloading from existing API
        loadPatientData();
    });
}

function loadSamplePatientData() {
    console.log('📋 Loading sample patient data as fallback...');
    
    // Show notification that we're using sample data
    showError('API server not available - Using sample data for demonstration only');
    
    // Sample patient data for demonstration (only when API is down)
    patients = [
        {
            patient_id: 'P001',
            name: 'John Smith',
            age: 45,
            gender: 'Male',
            insurer: 'BlueCross BlueShield',
            claim_amount: 2500.00,
            procedure_code: '99214',
            diagnosis_code: 'J06.9',
            service_date: '2025-07-10',
            status: 'Ready'
        },
        {
            patient_id: 'P002',
            name: 'Sarah Johnson',
            age: 32,
            gender: 'Female',
            insurer: 'Aetna',
            claim_amount: 1800.00,
            procedure_code: '99213',
            diagnosis_code: 'I10',
            service_date: '2025-07-09',
            status: 'Ready'
        },
        {
            patient_id: 'P003',
            name: 'Michael Brown',
            age: 58,
            gender: 'Male',
            insurer: 'United Healthcare',
            claim_amount: 3200.00,
            procedure_code: '99215',
            diagnosis_code: 'E11.9',
            service_date: '2025-07-08',
            status: 'Ready'
        },
        {
            patient_id: 'P004',
            name: 'Emily Davis',
            age: 28,
            gender: 'Female',
            insurer: 'Cigna',
            claim_amount: 1500.00,
            procedure_code: '99212',
            diagnosis_code: 'R50.9',
            service_date: '2025-07-07',
            status: 'Ready'
        },
        {
            patient_id: 'P005',
            name: 'Robert Wilson',
            age: 65,
            gender: 'Male',
            insurer: 'Medicare',
            claim_amount: 4100.00,
            procedure_code: '99215',
            diagnosis_code: 'J45.9',
            service_date: '2025-07-06',
            status: 'Ready'
        },
        {
            patient_id: 'P006',
            name: 'Lisa Anderson',
            age: 41,
            gender: 'Female',
            insurer: 'Humana',
            claim_amount: 2200.00,
            procedure_code: '99214',
            diagnosis_code: 'M79.3',
            service_date: '2025-07-05',
            status: 'Ready'
        }
    ];
    
    console.log(`✅ Loaded ${patients.length} sample patients`);
    renderPatientCards();
    
    // Also load some sample rejected claims for the third tab
    loadSampleRejectedClaims();
}

function loadSampleRejectedClaims() {
    // Sample rejected claims data
    rejectedClaims = [
        {
            patient_id: 'P007',
            patient_name: 'David Miller',
            claim_amount: 3500.00,
            insurer: 'Anthem',
            completion_time: new Date('2025-07-05'),
            reason: 'Prior authorization required - procedure not pre-approved'
        },
        {
            patient_id: 'P008',
            patient_name: 'Jennifer Taylor',
            claim_amount: 2800.00,
            insurer: 'Kaiser Permanente',
            completion_time: new Date('2025-07-04'),
            reason: 'Medical necessity not established - insufficient documentation'
        },
        {
            patient_id: 'P009',
            patient_name: 'William Garcia',
            claim_amount: 1900.00,
            insurer: 'Blue Cross',
            completion_time: new Date('2025-07-03'),
            reason: 'Incorrect diagnosis code - ICD-10 code mismatch'
        }
    ];
    
    updateRejectedClaimsList();
}

function renderPatientCards() {
    const container = document.querySelector('.patient-grid');
    if (!container) return;
    container.innerHTML = '';
    
    if (!patients || patients.length === 0) {
        container.innerHTML = `
            <div class="card p-6 text-center">
                <div class="text-sm">No patient data available. Please check your data source.</div>
            </div>
        `;
        return;
    }

    // Optional search filtering
    const searchValue = (document.getElementById('search-input')?.value || '').toLowerCase();
    const list = patients.filter(p => {
        if (!searchValue) return true;
        return (
            (p.name || '').toLowerCase().includes(searchValue) ||
            (p.patient_id || '').toLowerCase().includes(searchValue)
        );
    });

    list.forEach(patient => {
        // Check if this patient has an active claim
        const hasActiveClaim = activeClaims.some(claim => claim.patient_id === patient.patient_id);
        const isProcessed = processedClaims.some(claim => claim.patient_id === patient.patient_id);
        const isRejected = rejectedClaims.some(claim => claim.patient_id === patient.patient_id);
        
        let statusBadgeClass = 'status-ready', statusLabel = 'Ready';
        let actionButtons;
        
        if (hasActiveClaim) {
            statusBadgeClass = 'status-processing';
            statusLabel = 'Processing';
            actionButtons = `
                <button class="btn-apply-claim" disabled>
                    <svg class="btn-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"/>
                    </svg>
                    Processing...
                </button>
                <button class="btn-view-info" onclick="viewPatientInfo('${patient.patient_id}')">
                    <svg class="btn-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                    </svg>
                </button>
            `;
        } else if (isProcessed) {
            const processedClaim = processedClaims.find(claim => claim.patient_id === patient.patient_id);
            const success = processedClaim && processedClaim.status === 'approved';
            statusBadgeClass = success ? 'status-approved' : 'status-denied';
            statusLabel = success ? 'Approved' : 'Denied';
            actionButtons = `
                <button class="btn-apply-claim" onclick="viewClaimDetails && viewClaimDetails('${patient.patient_id}')">
                    <svg class="btn-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"/>
                    </svg>
                    View Details
                </button>
                <button class="btn-view-info" onclick="viewPatientInfo('${patient.patient_id}')">
                    <svg class="btn-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                    </svg>
                </button>
            `;
        } else {
            statusBadgeClass = 'status-ready';
            statusLabel = 'Ready';
            actionButtons = `
                <button class="btn-apply-claim" onclick="processClaim('${patient.patient_id}')">
                    <svg class="btn-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"/>
                    </svg>
                    Apply Claim
                </button>
                <button class="btn-view-info" onclick="viewPatientInfo('${patient.patient_id}')">
                    <svg class="btn-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                    </svg>
                </button>
            `;
        }
        // Apply status filter
        if (currentStatusFilter && currentStatusFilter !== 'All Status' && statusLabel !== currentStatusFilter) {
            return; // skip rendering this card
        }

        const initials = (patient.name || 'P').split(' ').map(n => n[0]).slice(0,2).join('');
        const patientCard = `
            <div class="patient-card">
                <div class="patient-card-header">
                    <div class="patient-info">
                        <div class="patient-avatar"><span>${initials}</span></div>
                        <div class="patient-details">
                            <h3>${patient.name || 'Unknown'}</h3>
                            <p>ID: ${patient.patient_id || '-'}</p>
                        </div>
                    </div>
                    <span class="status-badge ${statusBadgeClass}">${statusLabel}</span>
                </div>
                <div class="patient-card-body">
                    <div class="patient-field"><span class="field-label">Insurer</span><span class="field-value">${patient.insurer || '-'}</span></div>
                    <div class="patient-field"><span class="field-label">Service Date</span><span class="field-value">${patient.service_date || '-'}</span></div>
                    <div class="patient-field"><span class="field-label">Procedure</span><span class="field-value">${patient.procedure_code || '-'}</span></div>
                    <div class="patient-field"><span class="field-label">Diagnosis</span><span class="field-value">${patient.diagnosis_code || '-'}</span></div>
                    <div class="patient-field"><span class="field-label">Amount</span><span class="field-value amount">$${parseFloat(patient.claim_amount || 0).toLocaleString()}</span></div>
                </div>
                <div class="patient-card-footer">${actionButtons}</div>
            </div>
        `;
        
        container.innerHTML += patientCard;
    });
}

function processClaim(patientId) {
    const patient = patients.find(p => p.patient_id === patientId);
    if (!patient) {
        showError('Patient not found');
        return;
    }
    
    console.log(`🚀 Starting claim processing for patient: ${patient.name}`);
    
    // Add to active claims
    const activeClaim = {
        patient_id: patientId,
        patient_name: patient.name,
        claim_amount: patient.claim_amount,
        insurer: patient.insurer,
        start_time: new Date(),
        status: 'processing'
    };
    
    activeClaims.push(activeClaim);
    
    // Update metrics
    metrics.activeClaims = activeClaims.length;
    updateMetrics();
    
    // Show processing notification
    showSuccess(`Processing started for ${patient.name}`);
    
    // Start the agentic claim processing
    startAgenticProcessing(patient);
    
    // Refresh the patient cards
    renderPatientCards();
    
    // Update recent activity
    updateRecentActivity(`Started processing claim for ${patient.name}`);
}

function startAgenticProcessing(patient) {
    console.log('🤖 Starting agentic claim processing workflow...');
    
    // Submit to the backend agentic system
    const submitUrl = `${API_BASE_URL}/api/submit-claim`;
    fetch(submitUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            patient_id: patient.patient_id,
            patient_name: patient.name,
            procedure_code: patient.procedure_code,
            diagnosis_code: patient.diagnosis_code,
            claim_amount: patient.claim_amount,
            service_date: patient.service_date,
            insurer: patient.insurer,
            provider: patient.provider || 'Hospital',
            notes: `Claim for ${patient.name} - ${patient.diagnosis_code}`
        })
    })
    .then(response => response.json())
    .then(result => {
        console.log('✅ Agentic processing started:', result);
        
        if (result.success) {
            // Update the active claim with the claim ID
            const activeClaim = activeClaims.find(claim => claim.patient_id === patient.patient_id);
            if (activeClaim) {
                activeClaim.claim_id = result.claim_id;
                activeClaim.status = result.status;
            }
        } else {
        console.error('❌ Failed to start agentic processing:', result.error);
        showError('Failed to start claim processing: ' + result.error);
            
            // Remove from active claims
            activeClaims = activeClaims.filter(claim => claim.patient_id !== patient.patient_id);
            metrics.activeClaims = activeClaims.length;
            updateMetrics();
            renderPatientCards();
        }
    })
    .catch(error => {
        console.error('❌ Error starting agentic processing:', error);
        console.log('🔄 API not available, using simulation mode...');
        
        // Simulate the agentic processing when API is not available
        simulateAgenticProcessing(patient);
    });
}

function simulateAgenticProcessing(patient) {
        console.log('🎭 Simulating agentic workflow for', patient.name);
    
    // Update processing status messages via activity log
    setTimeout(() => updateRecentActivity('Risk assessment in progress...'), 1000);
    setTimeout(() => updateRecentActivity('Auto-correcting claim data...'), 2500);
    setTimeout(() => updateRecentActivity('Submitting to insurer...'), 4000);
    setTimeout(() => updateRecentActivity('Finalizing claim...'), 5500);
    
    setTimeout(() => {
        // Complete the claim processing
        const activeClaim = activeClaims.find(claim => claim.patient_id === patient.patient_id);
        if (activeClaim) {
            completeClaim(activeClaim);
        }
    }, 7000);
}

function checkActiveClaimsStatus() {
    if (activeClaims.length === 0) return;
    
    // Simulate claim processing completion (in real system, this would check the API)
    activeClaims.forEach(claim => {
        const processingTime = new Date() - claim.start_time;
        
        // Simulate 1-minute processing time
        if (processingTime > 60000) { // 1 minute
            completeClaim(claim);
        }
    });
}

function completeClaim(claim) {
    console.log(`✅ Claim processing completed for ${claim.patient_name}`);
    
    // Simulate success/failure (in real system, this would come from the API)
    const success = Math.random() > 0.3; // 70% success rate
    
    // Generate specific denial info if claim is rejected
    const denialInfo = !success ? generateDenialInfo(claim) : null;
    
    const processedClaim = {
        ...claim,
        status: success ? 'approved' : 'denied',
        completion_time: new Date(),
        processed_amount: success ? parseFloat(claim.claim_amount) : 0,
        reason: success ? 'Claim approved successfully' : denialInfo.reason,
        denial_info: denialInfo
    };
    
    // Move from active to processed or rejected
    activeClaims = activeClaims.filter(c => c.patient_id !== claim.patient_id);
    
    if (success) {
        processedClaims.push(processedClaim);
        metrics.successClaims += 1;
        metrics.recoveredAmount += processedClaim.processed_amount;
    } else {
        rejectedClaims.push(processedClaim);
        updateRejectedClaimsList();
    }
    
    // Update metrics
    metrics.activeClaims = activeClaims.length;
    metrics.claimsApplied += 1;
    metrics.successRate = metrics.claimsApplied > 0 ? Math.round((metrics.successClaims / metrics.claimsApplied) * 100) : 0;
    
    updateMetrics();
    renderPatientCards();
    
    // Update recent activity
    updateRecentActivity(`Claim for ${claim.patient_name} ${success ? 'approved' : 'denied'}`);
    
    // Show success notification
    showSuccess(`Claim for ${claim.patient_name} ${success ? 'approved' : 'denied'}`);
}

function generateDenialInfo(claim) {
    // Define specific denial patterns for each insurer
    const denialPatterns = {
        'BlueCross': [
            {
                reason: 'Prior Authorization Expired',
                details: 'Authorization expired or not valid for service date',
                requirements: [
                    'Updated prior authorization form',
                    'Clinical documentation supporting medical necessity',
                    'Updated service dates'
                ],
                success_rate: 0.75
            },
            {
                reason: 'Service Level Mismatch',
                details: 'Billed service level exceeds authorized level of care',
                requirements: [
                    'Updated level of care documentation',
                    'Clinical justification for service level',
                    'Provider credentials for service level'
                ],
                success_rate: 0.80
            }
        ],
        'Aetna': [
            {
                reason: 'Diagnosis Code Specificity',
                details: 'ICD-10 code requires higher specificity for procedure',
                requirements: [
                    'Updated diagnosis codes',
                    'Clinical notes supporting diagnosis',
                    'Recent examination findings'
                ],
                success_rate: 0.85
            },
            {
                reason: 'Medical Record Documentation',
                details: 'Insufficient clinical documentation for service provided',
                requirements: [
                    'Complete progress notes',
                    'Relevant test results',
                    'Treatment plan documentation'
                ],
                success_rate: 0.90
            }
        ],
        'Cigna': [
            {
                reason: 'Medical Necessity Criteria',
                details: 'Documentation does not meet medical necessity guidelines',
                requirements: [
                    'Clinical findings documentation',
                    'Failed conservative treatment history',
                    'Objective measurement data'
                ],
                success_rate: 0.70
            },
            {
                reason: 'Treatment Plan Documentation',
                details: 'Incomplete or missing treatment plan documentation',
                requirements: [
                    'Detailed treatment goals',
                    'Expected outcomes',
                    'Treatment frequency and duration'
                ],
                success_rate: 0.85
            }
        ],
        'United': [
            {
                reason: 'Provider Network Status',
                details: 'Provider credentials require verification for service',
                requirements: [
                    'Updated provider credentialing',
                    'Network participation verification',
                    'Facility accreditation documentation'
                ],
                success_rate: 0.80
            },
            {
                reason: 'Procedure Code Documentation',
                details: 'Procedure documentation incomplete for billed code',
                requirements: [
                    'Complete procedure notes',
                    'Supporting clinical indicators',
                    'Time documentation for timed codes'
                ],
                success_rate: 0.85
            }
        ]
    };

    // Get denial patterns for this insurer
    const insurerPatterns = denialPatterns[claim.insurer] || [];
    
    // If no patterns found for this insurer, return generic denial
    if (insurerPatterns.length === 0) {
        return {
            reason: 'Documentation Requirements Not Met',
            details: 'Additional documentation needed for claim processing',
            requirements: [
                'Complete medical records',
                'Updated claim form',
                'Supporting clinical documentation'
            ],
            success_rate: 0.60
        };
    }

    // Randomly select one of the denial patterns for this insurer
    return insurerPatterns[Math.floor(Math.random() * insurerPatterns.length)];
}

function viewPatientInfo(patientId) {
    const patient = patients.find(p => p.patient_id === patientId);
    if (!patient) {
        showError('Patient not found');
        return;
    }
    
    // Minimal info view using alert (no Bootstrap in new UI)
    alert(
        `Name: ${patient.name}\n` +
        `Patient ID: ${patient.patient_id}\n` +
        `Age: ${patient.age || '-'}  Gender: ${patient.gender || '-'}\n` +
        `Insurer: ${patient.insurer || '-'}\n` +
        `Procedure: ${patient.procedure_code || '-'}\n` +
        `Diagnosis: ${patient.diagnosis_code || '-'}\n` +
        `Amount: $${parseFloat(patient.claim_amount || 0).toLocaleString()}\n` +
        `Service Date: ${patient.service_date || '-'}\n` +
        `Medical History: ${patient.medical_history || 'N/A'}`
    );
}

function updateRecentActivity(activity) {
    // Legacy function for backward compatibility
    updateAgentActivity([{
        id: Date.now(),
        message: activity,
        agent: 'system',
        status: 'info',
        timestamp: new Date().toISOString(),
        duration: 0
    }]);
}

function updateViewModeText() {
    const viewModeText = document.getElementById('view-mode-text');
    if (viewModeText) {
        viewModeText.textContent = userFriendlyView ? 'User View' : 'Tech View';
    }
}

async function fetchAgentActivity() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/agent-activity`);
        const data = await response.json();
        
        if (data.success && data.activities) {
            updateAgentActivity(data.activities);
        }
    } catch (error) {
        console.error('Error fetching agent activity:', error);
        // Show error state
        updateAgentActivity([{
            id: 'error',
            message: '⚠️ Unable to connect to agent activity feed',
            agent: 'system',
            status: 'error',
            timestamp: new Date().toISOString(),
            duration: 0
        }]);
    }
}

function updateAgentActivity(activities) {
    const container = document.getElementById('recentActivity');
    
    // Store activities for view switching
    lastActivities = activities;
    
    if (!activities || activities.length === 0) {
        container.innerHTML = `
            <div class="flex items-center space-x-3 p-3 bg-muted rounded-lg">
                <div class="h-4 w-4 bg-gray-300 rounded-full"></div>
                <span class="text-muted-foreground">No agent activity available</span>
            </div>
        `;
        return;
    }
    
    const activitiesHtml = activities.map(activity => {
        const timeAgo = getTimeAgo(activity.timestamp);
        const statusColor = getStatusColor(activity.status);
        const agentIcon = getAgentIcon(activity.agent);
        const isActive = activity.status === 'processing';
        
        // Choose content based on view mode
        let displayMessage, displayDetails;
        if (userFriendlyView && activity.user_friendly_activity) {
            displayMessage = activity.user_friendly_activity;
            displayDetails = activity.user_friendly_details || '';
        } else {
            displayMessage = activity.activity || activity.message || 'Processing step';
            displayDetails = activity.details || '';
        }
        
        const nextSteps = activity.next_steps || '';
        const patientContext = activity.patient_context || '';
        const hasTranslation = activity.has_translation;
        
        const agentName = activity.agent || 'System';
        const patientId = activity.patient_id || activity.patient_name || '';
        const category = activity.category || 'general';
        
        // Get patient name from ID if possible (extract from PAT### format)
        let patientDisplay = patientId;
        if (patientId.startsWith('PAT')) {
            patientDisplay = `Patient ${patientId.substring(3)}`;
        }
        
        return `
            <div class="flex items-start space-x-3 p-4 border-l-4 ${statusColor.border} bg-gradient-to-r from-gray-50 to-white rounded-r-lg hover:shadow-md transition-all duration-200 border border-gray-100">
                <div class="flex-shrink-0 mt-1">
                    ${isActive ? 
                        `<div class="animate-spin h-5 w-5 border-2 border-primary border-t-transparent rounded-full"></div>` :
                        `<div class="h-5 w-5 ${statusColor.bg} rounded-full flex items-center justify-center shadow-sm">
                            <span class="text-xs text-white font-medium">${agentIcon}</span>
                        </div>`
                    }
                </div>
                <div class="flex-1 min-w-0">
                    <div class="flex items-start justify-between mb-2">
                        <div class="flex-1">
                            <p class="text-sm font-semibold text-gray-900 leading-tight">${displayMessage}</p>
                            ${patientDisplay ? 
                                `<p class="text-xs font-medium text-blue-600 mt-1">👤 ${patientDisplay}</p>` : ''
                            }
                        </div>
                        <div class="flex items-center space-x-2 ml-3">
                            ${userFriendlyView && hasTranslation ? 
                                `<span class="text-xs px-2 py-1 rounded-full bg-green-100 text-green-700 font-medium" title="AI-translated to user-friendly language">✨ Friendly</span>` : 
                                !userFriendlyView ? `<span class="text-xs px-2 py-1 rounded-full bg-gray-100 text-gray-600 font-medium" title="Technical view">🔧 Tech</span>` : ''
                            }
                            <span class="text-xs px-2 py-1 rounded-full bg-blue-100 text-blue-800 font-medium">${agentName}</span>
                        </div>
                    </div>
                    ${displayDetails ? 
                        `<p class="text-sm text-gray-700 leading-relaxed mb-2">${displayDetails}</p>` : ''
                    }
                    ${userFriendlyView && patientContext ? 
                        `<p class="text-xs text-gray-500 italic mb-2">ℹ️ ${patientContext}</p>` : ''
                    }
                    <div class="flex items-center space-x-4 text-xs text-gray-500">
                        <span class="font-medium">${timeAgo}</span>
                        ${activity.duration > 0 ? 
                            `<span class="flex items-center space-x-1"><span>⏱️</span><span>${activity.duration}s</span></span>` : ''
                        }
                        ${category && category !== 'general' ? 
                            `<span class="px-2 py-1 rounded bg-gray-200 text-gray-700 font-medium">${category.replace('_', ' ')}</span>` : ''
                        }
                    </div>
                </div>
                ${isActive ? 
                    `<div class="flex-shrink-0">
                        <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    </div>` : ''
                }
            </div>
        `;
    }).join('');
    
    container.innerHTML = activitiesHtml;
}

function getTimeAgo(timestamp) {
    const now = new Date();
    const time = new Date(timestamp);
    const diffInSeconds = Math.floor((now - time) / 1000);
    
    if (diffInSeconds < 60) return `${diffInSeconds}s ago`;
    if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m ago`;
    if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}h ago`;
    return time.toLocaleDateString();
}

function getStatusColor(status) {
    const colors = {
        'processing': { bg: 'bg-blue-500', border: 'border-blue-500' },
        'completed': { bg: 'bg-green-500', border: 'border-green-500' },
        'approved': { bg: 'bg-green-600', border: 'border-green-600' },
        'submitted': { bg: 'bg-yellow-500', border: 'border-yellow-500' },
        'appeal_created': { bg: 'bg-purple-500', border: 'border-purple-500' },
        'error': { bg: 'bg-red-500', border: 'border-red-500' },
        'info': { bg: 'bg-gray-500', border: 'border-gray-500' }
    };
    return colors[status] || colors['info'];
}

function getAgentIcon(agent) {
    const icons = {
        'Risk Predictor': '🧠',
        'Auto Corrector': '🔧', 
        'Claim Submitter': '📤',
        'Appeal Generator': '📝',
        'Resubmitter': '🔄',
        'Feedback Learner': '📈',
        'System': 'ℹ️',
        'RiskPredictor': '🧠',
        'AutoCorrector': '🔧', 
        'ClaimSubmitter': '📤',
        'AppealGenerator': '📝',
        'FeedbackLearner': '📈',
        // Legacy support
        'risk_predictor': '🧠',
        'auto_corrector': '🔧', 
        'claim_submitter': '📤',
        'appeal_generator': '📝',
        'resubmitter': '🔄',
        'feedback_learner': '📈',
        'completed': '✅',
        'system': 'ℹ️'
    };
    return icons[agent] || '⚙️';
}

// Auto-refresh agent activity every 3 seconds
let activityRefreshInterval;

function startActivityRefresh() {
    // Initial fetch
    fetchAgentActivity();
    
    // Set up auto-refresh
    if (activityRefreshInterval) {
        clearInterval(activityRefreshInterval);
    }
    
    activityRefreshInterval = setInterval(fetchAgentActivity, 3000);
}

function stopActivityRefresh() {
    if (activityRefreshInterval) {
        clearInterval(activityRefreshInterval);
        activityRefreshInterval = null;
    }
}

function updateRejectedClaimsList() {
    const container = document.getElementById('rejectedClaimsList');
    
    // Update heading to show count
    const heading = document.querySelector('.rejected-claims-heading');
    if (heading) {
        heading.textContent = `Rejected Claims (${rejectedClaims.length})`;
    }
    
    if (rejectedClaims.length === 0) {
        container.innerHTML = '<p class="text-muted-foreground">No rejected claims - AI system automatically handles all rejections</p>';
        return;
    }

    const sorted = [...rejectedClaims].sort((a, b) => {
        const dateA = new Date(a.rejection_date || a.completion_time);
        const dateB = new Date(b.rejection_date || b.completion_time);
        return dateB - dateA;
    });

    const html = sorted.map(claim => {
        const denial = claim.denial_info || {};
        const amount = `$${parseFloat(claim.claim_amount || 0).toLocaleString()}`;
        const reason = denial.reason || claim.reason || 'Rejected';
        
        // Determine if additional patient data is needed in OpenEMR
        const requiredItems = denial.requirements || [];
        const needsPatientUpdate = requiredItems.some(item => 
            item.toLowerCase().includes('medical history') || 
            item.toLowerCase().includes('documentation') ||
            item.toLowerCase().includes('authorization') ||
            item.toLowerCase().includes('clinical')
        );
        
        return `
            <div class="border border-destructive-20 rounded-lg p-4 bg-destructive-5">
                <div class="flex items-center justify-between mb-3">
                    <h3 class="font-semibold">${claim.patient_name || claim.patient_id}</h3>
                    <div class="badge-destructive">${amount}</div>
                </div>
                <p class="text-sm text-muted-foreground mb-3">${reason}</p>
                
                ${needsPatientUpdate ? `
                    <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-3 mb-3">
                        <div class="flex items-center space-x-2 mb-2">
                            <svg class="h-4 w-4 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                            </svg>
                            <span class="text-sm font-medium text-yellow-800">Patient Data Update Required</span>
                        </div>
                        <p class="text-xs text-yellow-700 mb-2">Please update the following in OpenEMR patient records:</p>
                        <ul class="text-xs text-yellow-700 list-disc list-inside space-y-1">
                            ${requiredItems.map(item => `<li>${item}</li>`).join('')}
                        </ul>
                        <div class="mt-2 flex items-center space-x-2">
                            <button class="btn-sm bg-yellow-100 text-yellow-800 border border-yellow-300 hover:bg-yellow-200" 
                                    onclick="openPatientInOpenEMR('${claim.patient_id}')">
                                <svg class="h-3 w-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-2"/>
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 3l-9 9"/>
                                </svg>
                                Update in OpenEMR
                            </button>
                            <span class="text-xs text-yellow-600">Once updated, AI will auto-retry claim</span>
                        </div>
                    </div>
                ` : `
                    <div class="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-3">
                        <div class="flex items-center space-x-2">
                            <svg class="h-4 w-4 text-blue-600 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"/>
                            </svg>
                            <span class="text-sm font-medium text-blue-800">AI Auto-Processing</span>
                        </div>
                        <p class="text-xs text-blue-700 mt-1">Appeal generated automatically - AI agents handling resubmission</p>
                    </div>
                `}
                
                <div class="flex items-center space-x-2">
                    <button class="btn-outline btn-sm" onclick="viewClaimDetails('${claim.patient_id}')">
                        <svg class="h-3 w-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                        </svg>
                        View Details
                    </button>
                    <button class="btn-outline btn-sm" onclick="viewAppealStatus('${claim.patient_id}')">
                        <svg class="h-3 w-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/>
                        </svg>
                        Appeal Status
                    </button>
                </div>
            </div>
        `;
    }).join('');

    container.innerHTML = html;
}

function refreshData() {
    console.log('🔄 Refreshing dashboard data from OpenEMR database...');
    refreshDataFromCSV();
}

function showSuccess(message) {
    showToast(message, 'success');
}

function showError(message) {
    showToast(message, 'error');
}

function showToast(message, type = 'success') {
    const wrap = document.createElement('div');
    wrap.style.position = 'fixed';
    wrap.style.top = '1rem';
    wrap.style.right = '1rem';
    wrap.style.zIndex = '9999';
    wrap.style.maxWidth = '320px';
    wrap.style.pointerEvents = 'none';

    const card = document.createElement('div');
    card.className = 'card shadow-soft';
    card.style.borderLeft = type === 'success' ? '4px solid hsl(142 76% 36%)' : '4px solid hsl(0 84% 60%)';
    card.style.padding = '12px 14px';
    card.style.marginBottom = '8px';
    card.innerText = message;

    wrap.appendChild(card);
    document.body.appendChild(wrap);

    setTimeout(() => {
        if (document.body.contains(wrap)) document.body.removeChild(wrap);
    }, 3000);
}

// Reference-like animations
function animateTabChange() {
    const activeContent = document.querySelector('.tab-content.active');
    if (!activeContent) return;
    activeContent.style.opacity = '0';
    activeContent.style.transform = 'translateY(10px)';
    activeContent.style.transition = 'all 0.3s ease';
    requestAnimationFrame(() => {
        setTimeout(() => {
            activeContent.style.opacity = '1';
            activeContent.style.transform = 'translateY(0)';
        }, 50);
    });
}

function animateMetricCards() {
    const cards = document.querySelectorAll('#records-tab .metrics-container > .metric-card');
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'all 0.5s ease';
        setTimeout(() => {
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, 100 + index * 150);
    });
}

function animateActivityFeed() {
    const items = document.querySelectorAll('#records-tab #recentActivity > *');
    items.forEach((item, index) => {
        item.style.opacity = '0';
        item.style.transform = 'translateX(-20px)';
        item.style.transition = 'all 0.3s ease';
        setTimeout(() => {
            item.style.opacity = '1';
            item.style.transform = 'translateX(0)';
        }, 100 + index * 200);
    });
}

// Export functions for testing
window.processClaim = processClaim;
window.viewPatientInfo = viewPatientInfo;
window.refreshData = refreshData;

// New functions for improved rejection handling
function openPatientInOpenEMR(patientId) {
    // This would integrate with OpenEMR system
    console.log(`Opening patient ${patientId} in OpenEMR for data update`);
    
    // For demonstration, show what needs to be updated
    const claim = rejectedClaims.find(c => c.patient_id === patientId);
    if (claim && claim.denial_info && claim.denial_info.requirements) {
        const requirements = claim.denial_info.requirements.join('\n• ');
        alert(`Patient ${patientId} requires these updates in OpenEMR:\n\n• ${requirements}\n\nOnce updated, the AI system will automatically retry the claim.`);
    } else {
        alert(`Opening patient ${patientId} in OpenEMR system...`);
    }
    
    // In a real implementation, this would:
    // 1. Open OpenEMR patient record
    // 2. Highlight required fields
    // 3. Set up callback for when data is updated
    // 4. Automatically trigger claim reprocessing
}

function viewAppealStatus(patientId) {
    console.log(`Viewing appeal status for patient ${patientId}`);
    
    const claim = rejectedClaims.find(c => c.patient_id === patientId);
    if (!claim) {
        showError('Claim not found');
        return;
    }
    
    // Show appeal processing status
    const appealInfo = `
Appeal Status for ${claim.patient_name || patientId}

Status: Appeal Generated ✓
Appeal Letter: Created by AI
Expected Success Rate: ${claim.denial_info?.success_rate ? (claim.denial_info.success_rate * 100).toFixed(0) : '75'}%

Processing Timeline:
✓ Claim Submitted
✓ Rejected - ${claim.reason}
✓ AI Appeal Generated
⏳ Auto-Resubmission in Progress

The AI system is automatically handling:
• Appeal letter generation
• Supporting documentation
• Optimal resubmission timing
• Follow-up communications

No manual intervention required.
    `.trim();
    
    alert(appealInfo);
}

// Provide a basic claim details viewer for processed claims
function viewClaimDetails(patientId) {
    const claim = processedClaims.find(c => c.patient_id === patientId) || rejectedClaims.find(c => c.patient_id === patientId);
    if (!claim) {
        showError('No claim details found for this patient');
        return;
    }
    
    const status = claim.status || 'unknown';
    const denialInfo = claim.denial_info || {};
    
    let detailsText = `Patient: ${claim.patient_name || patientId}\n`;
    detailsText += `Status: ${status}\n`;
    detailsText += `Insurer: ${claim.insurer || '-'}\n`;
    detailsText += `Amount: $${parseFloat(claim.claim_amount || 0).toLocaleString()}\n`;
    
    if (status === 'denied' || status === 'rejected') {
        detailsText += `\nRejection Reason: ${claim.reason || denialInfo.reason || 'N/A'}\n`;
        if (denialInfo.details) {
            detailsText += `Details: ${denialInfo.details}\n`;
        }
        if (denialInfo.requirements && denialInfo.requirements.length > 0) {
            detailsText += `\nRequired for Approval:\n`;
            denialInfo.requirements.forEach(req => {
                detailsText += `• ${req}\n`;
            });
        }
        if (denialInfo.success_rate) {
            detailsText += `\nExpected Success Rate: ${(denialInfo.success_rate * 100).toFixed(0)}%\n`;
        }
        detailsText += `\nAI Status: Auto-processing appeal and resubmission`;
    } else if (status === 'approved') {
        detailsText += `Approved Amount: $${parseFloat(claim.approved_amount || claim.claim_amount || 0).toLocaleString()}`;
    }
    
    alert(detailsText);
}

window.viewClaimDetails = viewClaimDetails;
window.openPatientInOpenEMR = openPatientInOpenEMR;
window.viewAppealStatus = viewAppealStatus;
