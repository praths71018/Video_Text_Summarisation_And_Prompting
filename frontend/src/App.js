// import React, { useState, useEffect } from 'react';
// import Sidebar from './components/Sidebar';
// import TextPrompt from './components/TextPrompt';
// import ChatWindow from './components/ChatWindow';

// const App = () => {
//   const [chats, setChats] = useState([]); // State to track all chats
//   const [currentChatId, setCurrentChatId] = useState(null); // State to track the current chat
//   const [isLoading, setIsLoading] = useState(false); // State to track loading
//   const [isUploading, setIsUploading] = useState(false); // State to track video upload status
//   const [data, setData] = useState([]);

//   const addMessage = (chatId, message) => {
//     setChats((prevChats) =>
//       prevChats.map((chat) =>
//         chat.id === chatId ? { ...chat, messages: [...chat.messages, message] } : chat
//       )
//     );
//   };

//   const handleNewChat = () => {
//     const newChat = { id: Date.now(), messages: [] };
//     setChats((prevChats) => [newChat, ...prevChats]);
//     setCurrentChatId(newChat.id);
//   };

//   const handleSelectChat = (chatId) => {
//     setCurrentChatId(chatId);
//   };

//   useEffect(() => {
//     fetch("/members").then(
//       res => res.json()
//     ).then(
//       data => {
//         setData(data);
//         console.log(data);
//       }
//     );
//   }, []);

//   const currentChat = chats.find((chat) => chat.id === currentChatId);

//   return (
//     <div className="app" style={{
//       minHeight: '100vh',
//       display: 'flex',
//       flexDirection: 'column',
//       background: 'rgb(3,0,31)',
//       color: 'white',
//       fontFamily: "'Poppins', sans-serif",
//     }}>
//       <div style={{ display: 'flex', flex: 1 }}>
//         <Sidebar
//           chats={chats}
//           onNewChat={handleNewChat}
//           onSelectChat={handleSelectChat}
//         />
//         <div className="main" style={{ flex: 8, display: 'flex', flexDirection: 'column', padding: '1rem' }}>
//           {currentChat && (
//             <ChatWindow
//               messages={currentChat.messages}
//               isLoading={isLoading}
//               isUploading={isUploading}
//             />
//           )}
//         </div>
//       </div>
//       {currentChat && (
//         <TextPrompt
//           onSendMessage={(message) => addMessage(currentChat.id, message)}
//           setIsLoading={setIsLoading}
//           setIsUploading={setIsUploading}
//         />
//       )}
//     </div>
//   );
// };

// export default App;

import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import TextPrompt from './components/TextPrompt';
import ChatWindow from './components/ChatWindow';

const App = () => {
  const [chats, setChats] = useState([]); // State to track all chats
  const [currentChatId, setCurrentChatId] = useState(null); // State to track the current chat
  const [isLoading, setIsLoading] = useState(false); // State to track loading
  const [isUploading, setIsUploading] = useState(false); // State to track video upload status
  const [data, setData] = useState([]);

  const addMessage = (chatId, message) => {
    setChats((prevChats) =>
      prevChats.map((chat) =>
        chat.id === chatId ? { ...chat, messages: [...chat.messages, message] } : chat
      )
    );
  };

  const handleNewChat = () => {
    const newChat = { id: Date.now(), messages: [], name: `Chat ${Date.now()}` };
    setChats((prevChats) => [newChat, ...prevChats]);
    setCurrentChatId(newChat.id);
  };

  const handleSelectChat = (chatId) => {
    setCurrentChatId(chatId);
  };

  const handleRenameChat = (chatId, newName) => {
    setChats((prevChats) =>
      prevChats.map((chat) =>
        chat.id === chatId ? { ...chat, name: newName } : chat
      )
    );
  };

  useEffect(() => {
    fetch("/members").then(
      res => res.json()
    ).then(
      data => {
        setData(data);
        console.log(data);
      }
    );
  }, []);

  const currentChat = chats.find((chat) => chat.id === currentChatId);

  return (
    <div className="app" style={{
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      background: 'rgb(3,0,31)',
      color: 'white',
      fontFamily: "'Poppins', sans-serif",
    }}>
      <div style={{ display: 'flex', flex: 1 }}>
        <Sidebar
          chats={chats}
          onNewChat={handleNewChat}
          onSelectChat={handleSelectChat}
          onRenameChat={handleRenameChat}
        />
        <div className="main" style={{ flex: 8, display: 'flex', flexDirection: 'column', padding: '1rem' }}>
          {currentChat && (
            <ChatWindow
              messages={currentChat.messages}
              isLoading={isLoading}
              isUploading={isUploading}
            />
          )}
        </div>
      </div>
      {currentChat && (
        <TextPrompt
          onSendMessage={(message) => addMessage(currentChat.id, message)}
          setIsLoading={setIsLoading}
          setIsUploading={setIsUploading}
          onRenameChat={(newName) => handleRenameChat(currentChat.id, newName)}
        />
      )}
    </div>
  );
};

export default App;