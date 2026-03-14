// Healthcare Claims Dashboard - Main JS

// Initialize Socket.IO connection with proper config
const socket = io({
    transports: ['websocket', 'polling'],
    reconnection: true,
    reconnectionAttempts: 5
});

// Global state
let currentSessionId = null;
let activeWorkflow = false;
let voiceActive = false;

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const workflowProgress = document.getElementById('workflowProgress');
const voiceInterface = document.getElementById('voiceInterface');
const decisionResult = document.getElementById('decisionResult');
const loadingOverlay = document.getElementById('loadingOverlay');

// Step elements (from index.html)
const steps = {
    1: { 
        icon: document.getElementById('step1Icon'), 
        title: document.getElementById('step1Title'), 
        message: document.getElementById('step1Message'), 
        step: document.getElementById('step1') 
    },
    2: { 
        icon: document.getElementById('step2Icon'), 
        title: document.getElementById('step2Title'), 
        message: document.getElementById('step2Message'), 
        step: document.getElementById('step2') 
    },
    3: { 
        icon: document.getElementById('step3Icon'), 
        title: document.getElementById('step3Title'), 
        message: document.getElementById('step3Message'), 
        step: document.getElementById('step3') 
    },
    4: { 
        icon: document.getElementById('step4Icon'), 
        title: document.getElementById('step4Title'), 
        message: document.getElementById('step4Message'), 
        step: document.getElementById('step4') 
    }
};

// Stats elements
const totalClaimsEl = document.getElementById('total-claims');
const approvedClaimsEl = document.getElementById('approved-claims');
const rejectedClaimsEl = document.getElementById('rejected-claims');
const openTicketsEl = document.getElementById('open-tickets');

// Voice elements
const voiceStatus = document.getElementById('voiceStatus');
const voiceTranscript = document.getElementById('voiceTranscript');

// Decision elements
const decisionBox = document.getElementById('decisionBox');
const decisionText = document.getElementById('decisionText');
const reasoningText = document.getElementById('reasoningText');

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    console.log('Dashboard initializing...');
    loadDashboardData();
    setupEventListeners();
    setupSocketListeners();
    
    // Add refresh button listener
    const refreshBtn = document.getElementById('refreshBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', loadDashboardData);
    }
    
    // Add sample claim button listener
    const sampleClaimBtn = document.getElementById('sampleClaimBtn');
    if (sampleClaimBtn) {
        sampleClaimBtn.addEventListener('click', runSampleClaim);
    }
});

// Setup event listeners
function setupEventListeners() {
    if (dropZone && fileInput) {
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', handleDragOver);
        dropZone.addEventListener('dragleave', handleDragLeave);
        dropZone.addEventListener('drop', handleDrop);
        fileInput.addEventListener('change', handleFileSelect);
    }
}

// Setup Socket.IO listeners
function setupSocketListeners() {
    socket.on('connect', () => {
        console.log('✅ Connected to server with ID:', socket.id);
        currentSessionId = socket.id;
        updateConnectionStatus('connected');
    });

    socket.on('disconnect', () => {
        console.log('❌ Disconnected from server');
        updateConnectionStatus('disconnected');
    });

    socket.on('connect_error', (error) => {
        console.error('Connection error:', error);
        updateConnectionStatus('error');
    });
    
    socket.on('workflow_start', (data) => {
        console.log('🚀 Workflow started:', data);
        activeWorkflow = true;
        if (loadingOverlay) loadingOverlay.style.display = 'flex';
        resetSteps();
        updateStep(1, 'active', `Processing claim ${data.current}/${data.total_claims}`);
    });
    
    socket.on('agent_update', handleAgentUpdate);
    socket.on('claim_decision', handleClaimDecision);
    socket.on('voice_start', handleVoiceStart);
    socket.on('voice_speaking', handleVoiceSpeaking);
    socket.on('voice_listening', handleVoiceListening);
    socket.on('voice_response', handleVoiceResponse);
    socket.on('voice_correction', handleVoiceCorrection);
    socket.on('ticket_created', handleTicketCreated);
    socket.on('workflow_complete', handleWorkflowComplete);
    socket.on('all_workflows_complete', handleAllWorkflowsComplete);
    socket.on('workflow_error', handleWorkflowError);
}

// Update connection status
function updateConnectionStatus(status) {
    const statusEl = document.getElementById('connection-status');
    if (statusEl) {
        statusEl.textContent = status;
        statusEl.className = status;
    }
}

// Load dashboard data
async function loadDashboardData() {
    try {
        const [statsRes, claimsRes, policiesRes, ticketsRes] = await Promise.all([
            fetch('/api/stats'),
            fetch('/api/claims'),
            fetch('/api/policy-updates'),
            fetch('/api/tickets')
        ]);
        
        const stats = await statsRes.json();
        const claims = await claimsRes.json();
        const policies = await policiesRes.json();
        const tickets = await ticketsRes.json();
        
        updateStats(stats);
        renderClaimsTable(claims);
        renderPolicyUpdates(policies);
        renderTickets(tickets);
        
    } catch (error) {
        console.error('Failed to load dashboard data:', error);
    }
}

// Update statistics
function updateStats(stats) {
    if (totalClaimsEl) totalClaimsEl.textContent = stats.total_claims || 0;
    if (approvedClaimsEl) approvedClaimsEl.textContent = stats.approved || 0;
    if (rejectedClaimsEl) rejectedClaimsEl.textContent = stats.rejected || 0;
    if (openTicketsEl) openTicketsEl.textContent = stats.open_tickets || 0;
}

// Render claims table
function renderClaimsTable(claims) {
    const tbody = document.getElementById('claimsTableBody');
    if (!tbody) return;
    
    if (!claims || claims.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6" class="px-6 py-4 text-center text-gray-500">
                    No claims found
                </td>
            </tr>
        `;
        return;
    }
    
    tbody.innerHTML = claims.map(claim => {
        const decisionClass = claim.decision === 'APPROVED' ? 'text-green-600 font-medium' : 
                             claim.decision === 'REJECTED' ? 'text-red-600 font-medium' : 'text-gray-600';
        
        return `
            <tr>
                <td class="px-6 py-3 text-sm text-gray-900">${claim.id || claim.claim_id || 'N/A'}</td>
                <td class="px-6 py-3 text-sm text-gray-600">${claim.patient_id || 'N/A'}</td>
                <td class="px-6 py-3 text-sm text-gray-600">${claim.procedure_code || 'N/A'}</td>
                <td class="px-6 py-3 text-sm ${decisionClass}">${claim.decision || 'PENDING'}</td>
                <td class="px-6 py-3 text-sm text-gray-600">${claim.confidence || 'N/A'}</td>
                <td class="px-6 py-3 text-sm text-gray-600">${claim.created_at ? new Date(claim.created_at).toLocaleDateString() : 'N/A'}</td>
            </tr>
        `;
    }).join('');
}

// Render policy updates
function renderPolicyUpdates(policies) {
    const container = document.getElementById('policyUpdates');
    if (!container) return;
    
    if (!policies || policies.length === 0) {
        container.innerHTML = '<p class="text-sm text-gray-500 text-center py-4">No recent policy updates</p>';
        return;
    }
    
    container.innerHTML = policies.map(policy => {
        const impactClass = policy.impact === 'high' ? 'text-red-600' :
                           policy.impact === 'medium' ? 'text-yellow-600' : 'text-green-600';
        
        return `
            <div class="border-l-2 border-blue-500 pl-3 py-2">
                <p class="text-sm font-medium">${policy.policy_code || 'UNKNOWN'}</p>
                <p class="text-xs text-gray-600 mb-1">${policy.change_description || 'No description'}</p>
                <div class="flex items-center justify-between">
                    <span class="text-xs ${impactClass}">${policy.impact || 'unknown'} impact</span>
                    <span class="text-xs text-gray-500">${policy.detected_at ? new Date(policy.detected_at).toLocaleDateString() : 'N/A'}</span>
                </div>
            </div>
        `;
    }).join('');
}

// Render tickets
function renderTickets(tickets) {
    const container = document.getElementById('ticketsList');
    if (!container) return;
    
    if (!tickets || tickets.length === 0) {
        container.innerHTML = '<p class="text-sm text-gray-500 text-center py-4">No open tickets</p>';
        return;
    }
    
    container.innerHTML = tickets.map(ticket => `
        <div class="bg-yellow-50 rounded-lg p-3">
            <div class="flex items-start justify-between">
                <div>
                    <p class="text-sm font-medium">${ticket.id || 'N/A'}</p>
                    <p class="text-xs text-gray-600 mt-1">${ticket.issue || 'No issue'}</p>
                </div>
                <span class="px-2 py-1 bg-yellow-100 text-yellow-800 rounded-full text-xs">
                    ${ticket.priority || 'medium'}
                </span>
            </div>
        </div>
    `).join('');
}

// Handle Agent Updates
function handleAgentUpdate(data) {
    console.log('🤖 Agent update:', data);
    const { agent, status, message } = data;
    
    if (agent.includes('Agent 2')) {
        updateStep(1, status, message);
    } else if (agent.includes('Agent 3')) {
        updateStep(2, status, message);
    } else if (agent.includes('Agent 4')) {
        updateStep(3, status, message);
    } else if (agent.includes('Agent 5')) {
        updateStep(4, status, message);
    }
}

// Update step status
function updateStep(stepNum, status, message) {
    const step = steps[stepNum];
    if (!step || !step.step) return;
    
    step.step.classList.remove('active', 'complete');
    
    if (status === 'running') {
        step.step.classList.add('active');
        if (step.icon) {
            step.icon.className = 'step-icon active';
            step.icon.innerHTML = stepNum;
        }
    } else if (status === 'complete') {
        step.step.classList.add('complete');
        if (step.icon) {
            step.icon.className = 'step-icon complete';
            step.icon.innerHTML = '✓';
        }
    } else {
        step.step.classList.remove('active', 'complete');
        if (step.icon) {
            step.icon.className = 'step-icon pending';
            step.icon.innerHTML = stepNum;
        }
    }
    
    if (step.message) {
        step.message.textContent = message || 'Waiting...';
    }
}

// Handle Claim Decision
function handleClaimDecision(data) {
    console.log('📊 Claim decision:', data);
    if (!decisionResult) return;
    
    decisionResult.classList.remove('hidden');
    
    if (data.decision === 'APPROVED') {
        if (decisionBox) {
            decisionBox.className = 'p-4 rounded-lg bg-green-50 border-l-4 border-green-500';
        }
        if (decisionText) {
            decisionText.innerHTML = '<i class="fas fa-check-circle text-green-500 mr-2"></i> Claim Approved';
            decisionText.className = 'font-medium text-green-700';
        }
    } else {
        if (decisionBox) {
            decisionBox.className = 'p-4 rounded-lg bg-red-50 border-l-4 border-red-500';
        }
        if (decisionText) {
            decisionText.innerHTML = '<i class="fas fa-times-circle text-red-500 mr-2"></i> Claim Rejected';
            decisionText.className = 'font-medium text-red-700';
        }
    }
    
    if (reasoningText) {
        reasoningText.textContent = data.reasoning || 'No reasoning provided';
    }
}

// Handle Voice Start
function handleVoiceStart(data) {
    console.log('🎤 Voice started:', data);
    voiceActive = true;
    if (voiceInterface) {
        voiceInterface.classList.remove('hidden');
    }
    if (voiceStatus) {
        voiceStatus.textContent = data.message || 'Voice agent starting...';
    }
}

// Handle Voice Speaking
function handleVoiceSpeaking(data) {
    console.log('🔊 Speaking:', data);
    if (voiceStatus) {
        voiceStatus.textContent = 'Speaking...';
    }
    if (voiceTranscript) {
        voiceTranscript.textContent = `"${data.text || ''}"`;
    }
}

// Handle Voice Listening
function handleVoiceListening(data) {
    console.log('👂 Listening...');
    if (voiceStatus) {
        voiceStatus.textContent = data.message || 'Listening...';
    }
    if (voiceTranscript) {
        voiceTranscript.textContent = '';
    }
}

// Handle Voice Response
function handleVoiceResponse(data) {
    console.log('💬 Response:', data);
    if (voiceStatus) {
        voiceStatus.textContent = 'Response received';
    }
    if (voiceTranscript) {
        voiceTranscript.textContent = `You said: "${data.transcript || ''}" (Intent: ${data.intent || 'UNKNOWN'})`;
    }
}

// Handle Voice Correction
function handleVoiceCorrection(data) {
    console.log('🔄 Correction:', data);
    if (voiceStatus) {
        voiceStatus.textContent = data.message || 'Processing correction...';
    }
}

// Handle Ticket Created
function handleTicketCreated(data) {
    console.log('🎫 Ticket created:', data);
    loadDashboardData(); // Refresh tickets
    showNotification(`Ticket created: ${data.ticket_id}`, 'warning');
}

// Handle Workflow Complete
function handleWorkflowComplete(data) {
    console.log('✅ Workflow complete:', data);
    activeWorkflow = false;
    voiceActive = false;
    updateStep(4, 'complete', 'Workflow complete');
    if (voiceInterface) {
        voiceInterface.classList.add('hidden');
    }
    loadDashboardData();
    showNotification(`Workflow complete for claim ${data.claim_id}`, 'success');
    
    setTimeout(() => {
        if (loadingOverlay) {
            loadingOverlay.style.display = 'none';
        }
    }, 2000);
}

// Handle All Workflows Complete
function handleAllWorkflowsComplete(data) {
    console.log('🏁 All workflows complete:', data);
    setTimeout(() => {
        activeWorkflow = false;
        if (loadingOverlay) {
            loadingOverlay.style.display = 'none';
        }
    }, 2000);
}

// Handle Workflow Error
function handleWorkflowError(data) {
    console.error('❌ Workflow error:', data);
    activeWorkflow = false;
    voiceActive = false;
    if (voiceInterface) {
        voiceInterface.classList.add('hidden');
    }
    if (loadingOverlay) {
        loadingOverlay.style.display = 'none';
    }
    showNotification(`Error: ${data.error}`, 'error');
}

// Reset steps
function resetSteps() {
    for (let i = 1; i <= 4; i++) {
        updateStep(i, 'pending', 'Waiting to start...');
    }
    if (voiceInterface) {
        voiceInterface.classList.add('hidden');
    }
    if (decisionResult) {
        decisionResult.classList.add('hidden');
    }
}

// Handle Drag & Drop
function handleDragOver(e) {
    e.preventDefault();
    if (dropZone) {
        dropZone.classList.add('dragover');
    }
}

function handleDragLeave(e) {
    e.preventDefault();
    if (dropZone) {
        dropZone.classList.remove('dragover');
    }
}

function handleDrop(e) {
    e.preventDefault();
    if (dropZone) {
        dropZone.classList.remove('dragover');
    }
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        uploadFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        uploadFile(files[0]);
    }
}

async function uploadFile(file) {
    if (!file.name.endsWith('.json')) {
        showNotification('Please upload a JSON file', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            showNotification(data.error, 'error');
        } else {
            console.log(`📁 Upload successful:`, data);
            showNotification(`Processing ${data.claims} claim(s)...`, 'info');
            
            // Join the session room
            if (data.session_id) {
                socket.emit('join_session', { session_id: data.session_id });
            }
            
            resetWorkflowProgress();
            activeWorkflow = true;
        }
        
    } catch (error) {
        console.error('Upload failed:', error);
        showNotification('Upload failed', 'error');
    }
}

// Run sample claim
async function runSampleClaim() {
    const sampleClaim = {
        patient_id: "pat001",
        code: "J3420",
        dose: 1500,
        diagnosis: "E53.8"
    };
    
    const formData = new FormData();
    const blob = new Blob([JSON.stringify(sampleClaim)], { type: 'application/json' });
    const file = new File([blob], 'sample_claim.json', { type: 'application/json' });
    
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            showNotification(data.error, 'error');
        } else {
            showNotification('Sample claim uploaded', 'success');
            
            // Join the session room
            if (data.session_id) {
                socket.emit('join_session', { session_id: data.session_id });
            }
        }
        
    } catch (error) {
        console.error('Failed to run sample claim:', error);
        showNotification('Failed to run sample claim', 'error');
    }
}

// Show notification
function showNotification(message, type = 'info') {
    console.log(`[${type}] ${message}`);
    // You can implement a toast notification here
}

// Refresh data periodically
setInterval(loadDashboardData, 30000); // Every 30 seconds