import React from 'react';
import gptLogo from '../assets/gear.svg';
import addBtn from '../assets/add-30.png';
import msgIcon from '../assets/message.svg';
import home from '../assets/home.svg';
import saved from '../assets/bookmark.svg';
import rocket from '../assets/rocket.svg';

const Sidebar = ({ chats, onNewChat, onSelectChat }) => {
  // Inline styles
  const sideBarStyle = {
    flex: 2, // Adjusted width for a narrower sidebar
    borderRight: '1px solid rgb(200, 200, 200)',
    display: 'flex',
    flexDirection: 'column',
    minWidth: '12rem', // Set a minimum width for better alignment
  };

  const upperSideStyle = {
    padding: '1.5rem', // Adjusted padding for compactness
    borderBottom: '1px solid rgb(200, 200, 200)',
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
  };

  const upperSideTopStyle = {
    display: 'flex',
    alignItems: 'center',
    marginBottom: '1.5rem',
  };

  const addBtnStyle = {
    height: '1.5rem', // Smaller button size
    paddingRight: '0.5rem',
  };

  const midBtnStyle = {
    background: '#5A4BFF',
    border: 'none',
    padding: '1rem',
    fontSize: '1.2rem', // Smaller font size
    width: '100%', // Full width
    maxWidth: '10rem', // Restrict width for better fit
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    margin: '0 auto',
    marginBottom: '1.5rem',
    borderRadius: '0.5rem',
    cursor: 'pointer',
  };

  const queryStyle = {
    display: 'flex',
    alignItems: 'center',
    background: 'transparent',
    color: 'rgba(222,222,222,1)',
    padding: '0.75rem', // Adjusted padding
    width: '100%', // Full width
    maxWidth: '10rem', // Restrict width
    margin: '0.5rem auto',
    borderRadius: '0.5rem',
    border: '1px solid rgba(98,98,98,1)',
    fontSize: '0.9rem', // Smaller font size
    cursor: 'pointer',
  };

  const queryImgStyle = {
    marginRight: '1rem', // Adjusted margin
    objectFit: 'cover',
    height: '1.5rem', // Smaller image size
  };

  const lowerSideStyle = {
    padding: '1.5rem',
    flex: 1,
  };

  const listItemsStyle = {
    margin: '0.5rem 0', // Adjusted margin
    display: 'flex',
    alignItems: 'center',
    fontSize: '1.1rem', // Smaller font size
    cursor: 'pointer',
  };

  const listitemsImgStyle = {
    margin: '0.5rem', // Adjusted margin
    paddingRight: '0.5rem',
    height: '1.25rem', // Smaller image size
  };

  const upperSideBottomStyle = {
    flex: 1,
    overflowY: 'auto', // Enable vertical scrolling
  };

  return (
    <div className="sideBar" style={sideBarStyle}>
      <div className='upperSide' style={upperSideStyle}>
        <div className="upperSideTop" style={upperSideTopStyle}>
          <img src={gptLogo} alt='Logo' style={{ marginRight: '0.5rem' }} />
          <span className='brand' style={{ fontSize: '1.5rem' }}>GlobLearn</span>
        </div>
        <button className='midBtn' style={midBtnStyle} onClick={onNewChat}>
          <img src={addBtn} alt='new chat' style={addBtnStyle} />New Chat
        </button>
        <div className='upperSideBottom' style={upperSideBottomStyle}>
          {chats.map((chat) => (
            <button
              key={chat.id}
              className='query'
              style={queryStyle}
              onClick={() => onSelectChat(chat.id)}
            >
              <img src={msgIcon} alt='Query' style={queryImgStyle} />
              {chat.name}
            </button>
          ))}
        </div>
      </div>
      <div className='lowerSide' style={lowerSideStyle}>
        <div className='listItems' style={listItemsStyle}>
          <img src={home} alt='Home' style={listitemsImgStyle} />Home
        </div>
        <div className='listItems' style={listItemsStyle}>
          <img src={saved} alt='Saved' style={listitemsImgStyle} />Saved
        </div>
        <div className='listItems' style={listItemsStyle}>
          <img src={rocket} alt='Rocket' style={listitemsImgStyle} />Rocket
        </div>
      </div>
    </div>
  );
};

export default Sidebar;