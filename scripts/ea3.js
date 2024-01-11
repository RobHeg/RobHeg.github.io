//ea3.js

/**********************event listeners************************************************/
//submit chat button
document.getElementById('send-button').addEventListener('click', sendMessage);

//submit chat via Enter (not Shift+Enter)
document.getElementById('chat-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

/*************************sending a message into chat**********************************/
function sendMessage() {
    var input = document.getElementById('chat-input');
    var chatSpacing = document.getElementById('chat-spacing');

    // Only send the message if the input is not empty
    if (input.value.trim() !== '') {
        // Create a new chat-history element
        var newChatHistory = document.createElement('div');
        newChatHistory.id = 'chat-history';
        newChatHistory.innerHTML = '<p>' + input.value + '</p>';

        // Add the new chat-history to the beginning of the chat-spacing
        chatSpacing.prepend(newChatHistory);

        // Scroll to the bottom of the chat-spacing
        chatSpacing.scrollTop = chatSpacing.scrollHeight;

        // Clear the input field
        input.value = '';

        // Set focus back to the input field
        input.focus();
    }
}

//ea3.js
fetch('https://robheg.github.io/scripts/oliver_twist.txt')
    .then(response => response.text())
    .then(data => {
        // Verwenden Sie die Daten hier
        console.log(data);
    })
    .catch(error => console.error('Fehler beim Laden der Datei:', error));