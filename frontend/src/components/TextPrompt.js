import React, { useState, useEffect } from 'react';
import attachmentIcon from '../assets/attachment-svgrepo-com.svg';
import arrowIcon from '../assets/send.svg';
import videoIcon from '../assets/bookmark.svg';
import io from 'socket.io-client';

const TextPrompt = ({ onSendMessage, setIsLoading, setIsUploading, onRenameChat }) => {
  const [file, setFile] = useState(null);
  const [text, setText] = useState("");
  const [language, setLanguage] = useState(""); // State to track selected language

  useEffect(() => {
    const socket = io('http://127.0.0.1:5000');
    socket.on('log', (data) => {
      setIsLoading(true);
    });

    return () => {
      socket.disconnect();
    };
  }, [setIsLoading]);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile && selectedFile.type.startsWith("video")) {
      setFile(selectedFile); // Just set the file without sending the message
      onRenameChat(selectedFile.name); // Rename the chat to the video file name
    }
  };

  const handleTextChange = (event) => {
    setText(event.target.value);
  };

  const handleClearFile = () => {
    setFile(null);
  };

  const handleLanguageChange = (lang) => {
    setLanguage(prevLang => (prevLang === lang ? "" : lang)); // Toggle the language selection
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (text.trim()) {
      onSendMessage({ type: 'user', content: text });
      setIsLoading(true);
      try {
        const response = await fetch('http://127.0.0.1:5000/summarize_query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: text, language }), // Include the language
        });

        const result = await response.json();
        console.log("Summarized query result:", result);

        onSendMessage({
          type: 'transcript',
          content: result.summarized_response || "No response available",
        });

        onSendMessage({
          type: 'transcript',
          content: result.translated_response || "No response available",
        });
      } catch (error) {
        console.error('Error in summarizing query:', error);
      }

      setText("");
      setIsLoading(false);
    }

    if (file) {
      onSendMessage({
        type: 'video',
        content: URL.createObjectURL(file),
      });

      const formData = new FormData();
      formData.append('video', file);

      try {
        setIsUploading(true);
        const uploadResponse = await fetch('http://127.0.0.1:5000/upload_video', {
          method: 'POST',
          body: formData,
        });

        const uploadResult = await uploadResponse.json();
        console.log("Video upload response:", uploadResult);

        if (uploadResult.video_path) {
          const summarizeResponse = await fetch('http://127.0.0.1:5000/summarize_transcript', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              video_path: uploadResult.video_path,
              language, // Include the selected language
            }),
          });

          const summarizeResult = await summarizeResponse.json();
          console.log("Summarized transcript result:", summarizeResult);

          onSendMessage({
            type: 'transcript',
            content: summarizeResult.summarized_transcript || "No summarized transcript available",
          });

          onSendMessage({
            type: 'transcript',
            content: summarizeResult.translated_text || "No translated text available",
          });
        }

        setIsUploading(false);
        setFile(null);
      } catch (error) {
        console.error('Error processing video:', error);
        setIsUploading(false);
      }
    }
  };

  return (
    <div
      className="textPrompt"
      style={{
        position: "fixed",
        bottom: 0,
        left: "19.7%",
        width: "79%",
        background: "#1a1a1a", // Slightly lighter than black for better contrast
        borderTop: "1px solid #444", // Subtle border for separation
        padding: "1rem", // Increased padding for spaciousness
        display: "flex",
        alignItems: "center",
        boxShadow: "0 -2px 6px rgba(0, 0, 0, 0.5)", // Slightly softer shadow
        zIndex: 1000,
        overflow: "visible",
      }}
    >
      <form
        onSubmit={handleSubmit}
        style={{
          display: "flex",
          alignItems: "flex-start", // Align elements at the top for consistency
          width: "100%",
          gap: "1rem", // Even spacing between elements
        }}
      >
        {/* Text Input and Options */}
        <div style={{ position: "relative", flex: 1, display: "flex", flexDirection: "column" }}>
          <textarea
            style={{
              width: "90%",
              padding: "0.75rem 3rem 0.75rem 2.5rem", // Padding adjusted for icon space
              borderRadius: "0.5rem",
              border: "1px solid #555", // Softer border color
              background: "#2a2a2a", // Dark gray for better readability
              color: "white",
              fontSize: "1rem",
              resize: "none",
              outline: "none", // Remove default focus outline
              boxShadow: "inset 0 1px 2px rgba(0, 0, 0, 0.5)", // Subtle shadow for depth
              height: file ? "50px" : "65px", // Adjust height based on file presence
            }}
            placeholder="Type your message here..."
            onChange={handleTextChange}
            value={text}
            rows={file ? 2 : 3} // Adjust rows based on file presence
          />
          <div
            style={{
              marginTop: "0.5rem",
              display: "flex",
              gap: "1rem",
              justifyContent: "flex-start",
            }}
          >
            {/* Language Selection */}
            <label style={{ color: "white", cursor: "pointer", display: "flex", alignItems: "center" }}>
              <input
                type="checkbox"
                checked={language === "Hindi"}
                onChange={() => handleLanguageChange("Hindi")}
                style={{
                  marginRight: "0.5rem",
                  transform: "scale(1.2)", // Larger checkbox for better visibility
                }}
              />
              Hindi
            </label>
            <label style={{ color: "white", cursor: "pointer", display: "flex", alignItems: "center" }}>
              <input
                type="checkbox"
                checked={language === "French"}
                onChange={() => handleLanguageChange("French")}
                style={{
                  marginRight: "0.5rem",
                  transform: "scale(1.2)", // Larger checkbox for better visibility
                }}
              />
              French
            </label>
          </div>
          {/* Attachment Icon */}
          <img
            src={attachmentIcon}
            alt="Attach file"
            style={{
              position: "absolute",
              top: "50%",
              left: "1rem",
              transform: "translateY(-50%)",
              width: "1.5rem",
              height: "1.5rem",
              cursor: "pointer",
              opacity: 0.8, // Slight transparency for subtle effect
            }}
            onClick={() => document.getElementById("fileInput").click()}
          />
          <input
            type="file"
            id="fileInput"
            accept="video/*"
            onChange={handleFileChange}
            style={{ display: "none" }}
          />
        </div>

        {/* Video File Display */}
        {file && (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              background: "#444", // Background for contrast
              padding: "0.5rem",
              borderRadius: "0.5rem",
              maxWidth: "250px", // Limit width for cleaner look
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap", // Handle long file names
            }}
          >
            <img
              src={videoIcon}
              alt="Video attached"
              style={{ width: "1.5rem", height: "1.5rem", marginRight: "0.5rem" }}
            />
            <span style={{ color: "white", fontSize: "0.9rem", flex: 1 }}>{file.name}</span>
            <button
              type="button"
              onClick={handleClearFile}
              style={{
                background: "none",
                border: "none",
                color: "red",
                fontSize: "1rem",
                marginLeft: "0.5rem",
                cursor: "pointer",
              }}
            >
              ✕
            </button>
          </div>
        )}

        {/* Send Button */}
        <button
          type="submit"
          style={{
            background: "transparent",
            border: "none",
            cursor: "pointer",
            padding: "0.5rem",
          }}
        >
          <img src={arrowIcon} alt="Send" style={{ width: "2rem", height: "2rem", opacity: 0.9 }} />
        </button>
      </form>
    </div>
  );
};

export default TextPrompt;