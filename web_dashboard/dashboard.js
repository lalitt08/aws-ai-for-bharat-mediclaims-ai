/**
 * Healthcare Claims Processing Dashboard - Simple Tab Interface
 * 3 tabs: Patients, Records, Rejected Claims
 */

// Application state
let patients = [];
let activeClaims = [];
let processedClaims = [];
let rejectedClaims = [];
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

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    console.log('🏥 Healthcare Claims Processing Dashboard Loading...');
    console.log('🔍 Current URL:', window.location.href);
    console.log('🔍 Protocol:', window.location.protocol);
    console.log('🔍 Host:', window.location.host);
    console.log('🔍 Port:', window.location.port);
    
    // Check if elements exist
    const elements = ['patientsGrid', 'dataMode', 'activeClaimsCount', 'successClaimsCount'];
    elements.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            console.log(`✅ Element found: ${id}`);
        } else {
            console.error(`❌ Element not found: ${id}`);
        }
    });
    
    initializeDashboard();
});

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
    const container = document.getElementById('patientsGrid');
    container.innerHTML = `
        <div class="col-12 text-center py-5">
            <div class="spinner-border text-primary mb-3" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <h5>Loading Patient Data from CSV...</h5>
            <p class="text-muted">Reading real-time data from CSV files...</p>
        </div>
    `;
    
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
            document.getElementById('patientsGrid').innerHTML = `
                <div class="col-12 text-center py-5">
                    <div class="alert alert-warning">
                        <i class="fas fa-wifi me-2"></i>
                        <h5>Cannot connect to API server</h5>
                        <p>Make sure the server is running at ${API_BASE_URL}</p>
                        <button class="btn btn-primary" onclick="initializeDashboard()">
                            <i class="fas fa-refresh me-1"></i>Retry Connection
                        </button>
                    </div>
                </div>
            `;
            return;
        });
    
    // Initialize recent activity
    updateRecentActivity('Dashboard initialized - Loading CSV data...');
    
    // Set up auto-refresh for CSV data every 30 seconds
    setInterval(() => {
        console.log('🔄 Auto-refreshing CSV data...');
        refreshDataFromCSV();
    }, 30000); // Refresh every 30 seconds
    
    // Set up auto-refresh for active claims status every 5 seconds
    setInterval(checkActiveClaimsStatus, 5000); // Check every 5 seconds
}

function updateMetrics() {
    // Update Records tab metrics
    document.getElementById('activeClaimsCount').textContent = metrics.activeClaims;
    document.getElementById('successClaimsCount').textContent = metrics.successClaims;
    document.getElementById('recoveredAmountTotal').textContent = `$${metrics.recoveredAmount.toLocaleString()}`;
    document.getElementById('successRate').textContent = `${metrics.successRate}%`;
}

function loadPatientData() {
    console.log('📊 Loading patient data from CSV files...');
    console.log('🔍 Current window location:', window.location.href);
    const patientsUrl = `${API_BASE_URL}/api/patients`;
    console.log('🔍 Attempting to fetch from:', patientsUrl);
    
    // Clear previous error state
    document.getElementById('patientsGrid').innerHTML = `
        <div class="col-12 text-center py-5">
            <div class="spinner-border text-primary mb-3" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <h5>Loading Patient Data from CSV...</h5>
            <p class="text-muted">Fetching from ${patientsUrl}</p>
        </div>
    `;
    
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
                console.log(`✅ Loaded ${patients.length} patients from CSV`);
                console.log('📊 First patient:', patients[0]);
                renderPatientCards();
                document.getElementById('dataMode').innerHTML = `<span class="text-success">CSV Data (${patients.length} patients)</span>`;
                showSuccess(`Loaded ${patients.length} patients from CSV files`);
            } else if (Array.isArray(data) && data.length === 0) {
                console.warn('⚠️ Empty patient data array received');
                document.getElementById('patientsGrid').innerHTML = `
                    <div class="col-12 text-center py-5">
                        <div class="alert alert-warning">
                            <i class="fas fa-info-circle me-2"></i>
                            No patients found in CSV file.
                        </div>
                    </div>
                `;
            } else {
                console.error('❌ Invalid patient data format received:', data);
                document.getElementById('patientsGrid').innerHTML = `
                    <div class="col-12 text-center py-5">
                        <div class="alert alert-danger">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            Error: Invalid patient data format. Expected array, got ${typeof data}.
                        </div>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('❌ Error loading patient data:', error);
            console.error('❌ Error details:', error.message);
            console.error('❌ Error stack:', error.stack);
            
            // Display detailed error information
            document.getElementById('patientsGrid').innerHTML = `
                <div class="col-12 text-center py-5">
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        <h5>Failed to load patient data</h5>
                        <p><strong>Error:</strong> ${error.message}</p>
                        <p><strong>URL:</strong> ${API_BASE_URL}/api/patients</p>
                        <p><strong>Time:</strong> ${new Date().toLocaleString()}</p>
                        <hr>
                        <button class="btn btn-primary" onclick="loadPatientData()">
                            <i class="fas fa-refresh me-1"></i>Try Again
                        </button>
                        <button class="btn btn-secondary ms-2" onclick="location.reload()">
                            <i class="fas fa-redo me-1"></i>Reload Page
                        </button>
                    </div>
                </div>
            `;
        });
}

function refreshDataFromCSV() {
    console.log('🔄 Refreshing data from CSV files...');
    
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
    const container = document.getElementById('patientsGrid');
    container.innerHTML = '';
    
    if (!patients || patients.length === 0) {
        container.innerHTML = `
            <div class="col-12 text-center py-5">
                <div class="alert alert-info">
                    <i class="fas fa-info-circle me-2"></i>
                    No patient data available. Please check your data source.
                </div>
            </div>
        `;
        return;
    }
    
    patients.forEach(patient => {
        // Check if this patient has an active claim
        const hasActiveClaim = activeClaims.some(claim => claim.patient_id === patient.patient_id);
        const isProcessed = processedClaims.some(claim => claim.patient_id === patient.patient_id);
        const isRejected = rejectedClaims.some(claim => claim.patient_id === patient.patient_id);
        
        let statusBadge, actionButtons;
        
        if (hasActiveClaim) {
            statusBadge = '<span class="badge bg-warning">Processing</span>';
            actionButtons = `
                <button class="btn btn-sm btn-secondary me-2" disabled>
                    <i class="fas fa-clock me-1"></i>Processing...
                </button>
                <button class="btn btn-sm btn-outline-info" onclick="viewPatientInfo('${patient.patient_id}')">
                    <i class="fas fa-info-circle me-1"></i>View Info
                </button>
            `;
        } else if (isProcessed) {
            const processedClaim = processedClaims.find(claim => claim.patient_id === patient.patient_id);
            const success = processedClaim && processedClaim.status === 'approved';
            statusBadge = success ? 
                '<span class="badge bg-success">Approved</span>' : 
                '<span class="badge bg-danger">Denied</span>';
            actionButtons = `
                <button class="btn btn-sm btn-outline-success me-2" onclick="viewClaimDetails('${patient.patient_id}')">
                    <i class="fas fa-eye me-1"></i>View Details
                </button>
                <button class="btn btn-sm btn-outline-info" onclick="viewPatientInfo('${patient.patient_id}')">
                    <i class="fas fa-info-circle me-1"></i>View Info
                </button>
            `;
        } else {
            statusBadge = '<span class="badge bg-secondary">Ready</span>';
            actionButtons = `
                <button class="btn btn-sm btn-primary me-2" onclick="processClaim('${patient.patient_id}')">
                    <i class="fas fa-paper-plane me-1"></i>Apply Claim
                </button>
                <button class="btn btn-sm btn-outline-info" onclick="viewPatientInfo('${patient.patient_id}')">
                    <i class="fas fa-info-circle me-1"></i>View Info
                </button>
            `;
        }
        
        const patientCard = `
            <div class="col-md-4 mb-3">
                <div class="card patient-card">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-start mb-2">
                            <h6 class="card-title mb-0">${patient.name}</h6>
                            ${statusBadge}
                        </div>
                        <p class="card-text small text-muted mb-2">
                            <strong>ID:</strong> ${patient.patient_id}<br>
                            <strong>Insurer:</strong> ${patient.insurer}<br>
                            <strong>Amount:</strong> $${parseFloat(patient.claim_amount).toLocaleString()}
                        </p>
                        <div class="d-flex flex-column gap-2">
                            ${actionButtons}
                        </div>
                    </div>
                </div>
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
    
    // Show processing modal
    showProcessingModal(patient);
    
    // Start the agentic claim processing
    startAgenticProcessing(patient);
    
    // Refresh the patient cards
    renderPatientCards();
    
    // Update recent activity
    updateRecentActivity(`Started processing claim for ${patient.name}`);
}

function showProcessingModal(patient) {
    document.getElementById('processingPatientName').textContent = `Processing claim for ${patient.name}`;
    document.getElementById('processingStatus').textContent = 'Initializing agentic workflow...';
    
    const modal = new bootstrap.Modal(document.getElementById('processingModal'));
    modal.show();
    
    // Auto-hide modal after 5 seconds
    setTimeout(() => {
        modal.hide();
    }, 5000);
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
    
    // Update processing status messages
    setTimeout(() => {
        document.getElementById('processingStatus').textContent = 'Risk assessment in progress...';
    }, 1000);
    
    setTimeout(() => {
        document.getElementById('processingStatus').textContent = 'Auto-correcting claim data...';
    }, 2500);
    
    setTimeout(() => {
        document.getElementById('processingStatus').textContent = 'Submitting to insurer...';
    }, 4000);
    
    setTimeout(() => {
        document.getElementById('processingStatus').textContent = 'Finalizing claim...';
    }, 5500);
    
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
        
        // Simulate 3-minute processing time
        if (processingTime > 180000) { // 3 minutes
            completeClaim(claim);
        }
    });
}

function completeClaim(claim) {
    console.log(`✅ Claim processing completed for ${claim.patient_name}`);
    
    // Simulate success/failure (in real system, this would come from the API)
    const success = Math.random() > 0.3; // 70% success rate
    
    const processedClaim = {
        ...claim,
        status: success ? 'approved' : 'denied',
        completion_time: new Date(),
        processed_amount: success ? parseFloat(claim.claim_amount) : 0,
        reason: success ? 'Claim approved successfully' : 'Claim denied - insufficient documentation'
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

function updateRecentActivity(message) {
    const container = document.getElementById('recentActivity');
    const timestamp = new Date().toLocaleString();
    
    // Create new activity item
    const activityItem = document.createElement('div');
    activityItem.className = 'activity-item';
    activityItem.innerHTML = `
        <div class="activity-time">${timestamp}</div>
        <div class="activity-title">${message}</div>
    `;
    
    // Add to top of list
    if (container.firstChild && container.firstChild.tagName !== 'P') {
        container.insertBefore(activityItem, container.firstChild);
    } else {
        container.innerHTML = '';
        container.appendChild(activityItem);
    }
    
    // Keep only last 5 activities
    const activities = container.querySelectorAll('.activity-item');
    if (activities.length > 5) {
        activities[activities.length - 1].remove();
    }
}

function viewPatientInfo(patientId) {
    const patient = patients.find(p => p.patient_id === patientId);
    if (!patient) {
        showError('Patient not found');
        return;
    }
    
    const patientInfo = `
        <div class="row">
            <div class="col-md-6">
                <h6>Basic Information</h6>
                <p><strong>Name:</strong> ${patient.name}</p>
                <p><strong>Patient ID:</strong> ${patient.patient_id}</p>
                <p><strong>Age:</strong> ${patient.age}</p>
                <p><strong>Gender:</strong> ${patient.gender}</p>
                <p><strong>DOB:</strong> ${patient.dob}</p>
            </div>
            <div class="col-md-6">
                <h6>Claim Information</h6>
                <p><strong>Insurer:</strong> ${patient.insurer}</p>
                <p><strong>Procedure Code:</strong> ${patient.procedure_code}</p>
                <p><strong>Diagnosis Code:</strong> ${patient.diagnosis_code}</p>
                <p><strong>Claim Amount:</strong> $${parseFloat(patient.claim_amount).toLocaleString()}</p>
                <p><strong>Service Date:</strong> ${patient.service_date}</p>
            </div>
        </div>
        <div class="row mt-3">
            <div class="col-12">
                <h6>Medical History</h6>
                <p>${patient.medical_history || 'No medical history available'}</p>
            </div>
        </div>
    `;
    
    document.getElementById('patientInfoContent').innerHTML = patientInfo;
    const modal = new bootstrap.Modal(document.getElementById('patientInfoModal'));
    modal.show();
}

function updateRecentActivity(activity) {
    const container = document.getElementById('recentActivity');
    const timestamp = new Date().toLocaleString();
    
    const activityItem = `
        <div class="d-flex justify-content-between align-items-center mb-2">
            <span>${activity}</span>
            <small class="text-muted">${timestamp}</small>
        </div>
    `;
    
    if (container.querySelector('.text-muted')) {
        container.innerHTML = activityItem;
    } else {
        container.innerHTML = activityItem + container.innerHTML;
    }
    
    // Keep only last 5 activities
    const activities = container.querySelectorAll('.d-flex');
    if (activities.length > 5) {
        activities[activities.length - 1].remove();
    }
}

function updateRejectedClaimsList() {
    const container = document.getElementById('rejectedClaimsList');
    
    if (rejectedClaims.length === 0) {
        container.innerHTML = '<p class="text-muted">No rejected claims yet</p>';
        return;
    }
    
    container.innerHTML = rejectedClaims.map(claim => `
        <div class="card mb-3">
            <div class="card-body">
                <h6 class="card-title">
                    ${claim.patient_name}
                    <span class="badge bg-danger ms-2">REJECTED</span>
                </h6>
                <p class="card-text">
                    <strong>Claim Amount:</strong> $${parseFloat(claim.claim_amount).toLocaleString()}<br>
                    <strong>Insurer:</strong> ${claim.insurer}<br>
                    <strong>Rejected:</strong> ${claim.completion_time.toLocaleString()}
                </p>
                <p class="card-text">
                    <small class="text-danger">
                        <i class="fas fa-times-circle me-1"></i>
                        <strong>Reason:</strong> ${claim.reason}
                    </small>
                </p>
            </div>
        </div>
    `).join('');
}

function refreshData() {
    console.log('🔄 Refreshing dashboard data from CSV files...');
    refreshDataFromCSV();
}

function showSuccess(message) {
    const toast = document.createElement('div');
    toast.className = 'toast align-items-center text-white bg-success border-0 position-fixed top-0 end-0 m-3';
    toast.style.zIndex = '1050';
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                <i class="fas fa-check-circle me-2"></i>
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    document.body.appendChild(toast);
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
    
    setTimeout(() => {
        if (document.body.contains(toast)) {
            document.body.removeChild(toast);
        }
    }, 4000);
}

function showError(message) {
    const toast = document.createElement('div');
    toast.className = 'toast align-items-center text-white bg-danger border-0 position-fixed top-0 end-0 m-3';
    toast.style.zIndex = '1050';
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                <i class="fas fa-exclamation-triangle me-2"></i>
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    document.body.appendChild(toast);
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
    
    setTimeout(() => {
        if (document.body.contains(toast)) {
            document.body.removeChild(toast);
        }
    }, 4000);
}

// Export functions for testing
window.processClaim = processClaim;
window.viewPatientInfo = viewPatientInfo;
window.refreshData = refreshData;
