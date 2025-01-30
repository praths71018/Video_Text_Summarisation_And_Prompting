import React, { useEffect, useRef } from 'react';
import ChatMessage from './ChatMessage';
import { ShimmerButton } from 'react-shimmer-effects';

const ChatWindow = ({ messages, isLoading, isUploading }) => {
  const chatEndRef = useRef(null);

  useEffect(() => {
    // Scroll to the bottom every time a new message is added
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const chatWindowStyle = {
    display: 'flex',
    flexDirection: 'column',
    padding: '1rem',
    backgroundColor: '#03001F',
    borderRadius: '0.5rem',
    overflowY: 'auto',
    maxHeight: '70vh',
    marginBottom: '4rem',
  };

  return (
    <div style={chatWindowStyle}>
      {messages.map((msg, index) => (
        <ChatMessage key={index} type={msg.type} message={msg.content} />
      ))}
      {isUploading && (
        <div style={{ marginTop: "1rem" }}>
          <ShimmerButton size="xxl" style={{ borderRadius: "50%" }} />
        </div>
      )}
      <div ref={chatEndRef} /> {/* Empty div to anchor auto-scroll */}
    </div>
  );
};

export default ChatWindow;