import React from 'react';
import ReactMarkdown from 'react-markdown';
import gptLogo from '../assets/gear.svg';

const ChatMessage = ({ type, message }) => {
  const messageContainerStyle = {
    display: 'flex',
    margin: '0.5rem 0',
    justifyContent: type === 'user' ? 'flex-end' : 'flex-start',
    alignItems: 'flex-start',
  };

  const messageStyle = {
    maxWidth: '80%',
    padding: '0.75rem',
    borderRadius: '12px',
    wordBreak: 'break-word',
    display: 'inline-block',
    backgroundColor: type === 'user' ? '#5A4BFF' : '#2d2d2d',
    color: 'white',
    textAlign: type === 'user' ? 'right' : 'left',
  };

  return (
    <div style={messageContainerStyle}>
      {type === 'bot' && <img src={gptLogo} alt="GPT Logo" style={{ width: '35px', height: '35px', marginRight: '0.75rem' }} />}
      {type === 'video' ? (
        <video controls style={{ maxWidth: '100%' }}>
          <source src={message} type="video/mp4" />
          Your browser does not support the video tag.
        </video>
      ) : (
        // Render markdown if the type is 'transcript'
        <div style={messageStyle}>
          {type === 'transcript' ? (
            <ReactMarkdown>{message}</ReactMarkdown>
          ) : (
            message
          )}
        </div>
      )}
    </div>
  );
};

export default ChatMessage;