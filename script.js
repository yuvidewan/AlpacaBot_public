const form = document.getElementById('chat-form');
const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const databaseSelect = document.getElementById('database-select');

// const userAvatar = 'https://cdn-icons-png.flaticon.com/512/847/847969.png';
const userAvatar = "C:\\Users\\yuvra\\Desktop\\personal\\my pic.jpg"
const botAvatar = 'https://cdn-icons-png.flaticon.com/512/4712/4712109.png';

let currentDatabase = databaseSelect.value;
const toggleApiCheckbox = document.getElementById('toggle-api');

// Add welcome message on page load
document.addEventListener('DOMContentLoaded', () => {
    appendSystemMessage('Welcome to AlpacaBot! Ask me any data-driven questions about your database.');
    
    // Set up Gemini toggle animation
    const toggleApiCheckbox = document.getElementById('toggle-api');
    const apiToggleContainer = document.querySelector('.api-toggle');
    
    toggleApiCheckbox.addEventListener('change', function() {
        if (this.checked) {
            apiToggleContainer.setAttribute('data-active', 'true');
            apiToggleContainer.title = "Gemini brain activated! 🧠✨";
        } else {
            apiToggleContainer.setAttribute('data-active', 'false');
            apiToggleContainer.title = "Magic brain mode";
        }
    });
});

// Handle database change
databaseSelect.addEventListener('change', () => {
    const newDatabase = databaseSelect.value;
    if (newDatabase !== currentDatabase) {
        currentDatabase = newDatabase;
        appendSystemMessage(`Database changed to <strong>${newDatabase}</strong>`);
    }
});

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = userInput.value.trim();
    if (!message) return;

    const selectedDatabase = databaseSelect.value;
    const useApi = toggleApiCheckbox.checked;

    appendMessage('user', message, false);
    userInput.value = '';

    const typingEl = appendTyping();

    try {
        const response = await fetch('http://localhost:5000/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                database: selectedDatabase,
                use_api: useApi
            })
        });

        const data = await response.json();
        removeTyping(typingEl);

        if (response.ok) {
            let botResponse = '';
            if (data.generated_sentence && useApi) {
                botResponse += `<div class="bot-answer"><strong>Gemini:</strong><br>${data.generated_sentence}</div>`;
            }
            if (data.answer) {
                botResponse += `<div class="bot-answer"><strong>Answer:</strong><br>${data.answer}</div>`;
            }
            if (data.sql_query) {
                botResponse += `<div class="sql-query"><strong>SQL Query:</strong><br><code>${data.sql_query}</code></div>`;
            }
            if (data.confidence && data.confidence.length) {
                const confidenceStr = data.confidence.map(c => `${(c * 100).toFixed(1)}%`).join(', ');
                botResponse += `<div class="confidence"><strong>Confidence:</strong> ${confidenceStr}</div>`;
            }
            appendMessage('bot', botResponse, true);
        } else {
            showError(data.error || 'Something went wrong.', typingEl);
        }

    } catch (error) {
        console.error(error);
        showError('Backend unreachable. Please try again later.', typingEl);
    }
});

function appendMessage(sender, html, isHTML = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;

    const avatar = document.createElement('img');
    avatar.className = 'avatar';
    avatar.src = sender === 'user' ? userAvatar : botAvatar;
    avatar.alt = sender === 'user' ? 'User Avatar' : 'Bot Avatar';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    if (isHTML) {
        contentDiv.innerHTML = html;
    } else {
        contentDiv.textContent = html;
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function appendSystemMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message system';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = message;

    messageDiv.appendChild(contentDiv);
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function appendTyping() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot';

    const avatar = document.createElement('img');
    avatar.className = 'avatar';
    avatar.src = botAvatar;
    avatar.alt = 'Bot Avatar';

    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'message-content';
    typingIndicator.innerHTML = `
        <div class="typing-indicator">
            <span></span><span></span><span></span>
        </div>
    `;

    typingDiv.appendChild(avatar);
    typingDiv.appendChild(typingIndicator);
    chatBox.appendChild(typingDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    return typingDiv;
}

function removeTyping(typingElement) {
    if (typingElement && typingElement.parentElement) {
        typingElement.remove();
    }
}

function showError(message, typingElement) {
    removeTyping(typingElement);
    const errorHTML = `<div class="error-message"><strong>Error:</strong> ${message}</div>`;
    appendMessage('bot', errorHTML, true);
}

// Handle "Enter" key press for submitting the form
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        form.dispatchEvent(new Event('submit'));
    }
});