import React, { useState, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import '../styles/enhanced-chatbot.css';

/**
 * Enhanced Chatbot Component with Voice, Feedback, and Context Awareness
 * Features:
 * - Voice input/output
 * - Multi-language support
 * - Chat history
 * - Feedback system
 * - Follow-up questions for active learning
 */

const EnhancedChatbot = () => {
  const { t, i18n } = useTranslation();
  const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [followUpQuestions, setFollowUpQuestions] = useState([]);
  const [showFeedback, setShowFeedback] = useState(false);
  const [feedbackData, setFeedbackData] = useState(null);
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const messagesEndRef = useRef(null);
  const audioContextRef = useRef(null);
  const recognitionRef = useRef(null);

  useEffect(() => {
    // Check browser support for speech recognition
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      recognitionRef.current = new SpeechRecognition();
      setVoiceEnabled(true);
    }

    // Welcome message
    setMessages([
      {
        id: 1,
        text: t('chat.welcome') || 'Hello! I am your farming assistant. Ask me about pests, diseases, or farming practices.',
        sender: 'bot',
        timestamp: new Date()
      }
    ]);
  }, [t]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/assistant/enhanced-chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: localStorage.getItem('user_id') || 'user_' + Date.now(),
          query: userMessage.text,
          language: i18n.language
        })
      });

      const data = await response.json();

      const botMessage = {
        id: data.chat_id || Date.now(),
        text: data.response || 'Sorry, I could not process your request.',
        sender: 'bot',
        intent: data.intent,
        confidence: data.confidence,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);
      setFollowUpQuestions(data.follow_up_questions || []);

      if (data.response_audio_base64) {
        playAudioResponse(data.response_audio_base64);
      }

      setFeedbackData({
        chat_id: data.chat_id,
        response: data.response
      });

    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        id: Date.now(),
        text: t('chat.error') || 'Error: Could not reach the server. Please try again.',
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const playAudioResponse = async (audioBase64) => {
    if (!audioBase64) return;
    try {
      setIsPlayingAudio(true);
      const binaryString = atob(audioBase64);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      const audioContext = audioContextRef.current || new (window.AudioContext || window.webkitAudioContext)();
      audioContextRef.current = audioContext;

      const audioBuffer = await audioContext.decodeAudioData(bytes.buffer);
      const source = audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContext.destination);
      source.start(0);

      source.onended = () => {
        setIsPlayingAudio(false);
      };
    } catch (error) {
      console.error('Error playing audio:', error);
      setIsPlayingAudio(false);
    }
  };

  const toggleVoiceInput = () => {
    if (!recognitionRef.current) return;

    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    } else {
      recognitionRef.current.start();
      setIsListening(true);

      recognitionRef.current.onresult = (event) => {
        let interimTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            setInputValue(prev => prev + transcript + ' ');
          } else {
            interimTranscript += transcript;
          }
        }
      };

      recognitionRef.current.onerror = (event) => {
        console.error('Voice recognition error:', event.error);
        setIsListening(false);
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
      };
    }
  };

  const handleFollowUpQuestion = async (questionId, selectedAnswer) => {
    try {
      await fetch(`${API_BASE_URL}/api/assistant/feedback-answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: localStorage.getItem('user_id'),
          chat_id: feedbackData?.chat_id,
          question_id: questionId,
          answer: selectedAnswer
        })
      });

      setFollowUpQuestions(prev => prev.filter(q => q.id !== questionId));
    } catch (error) {
      console.error('Error submitting follow-up answer:', error);
    }
  };

  const handleMessageRating = async (rating, isHelpful) => {
    if (!feedbackData) return;

    try {
      await fetch(`${API_BASE_URL}/api/assistant/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chat_id: feedbackData.chat_id,
          is_helpful: isHelpful,
          user_rating: rating
        })
      });

      setShowFeedback(false);
      setFeedbackData(null);
    } catch (error) {
      console.error('Error submitting feedback:', error);
    }
  };

  return (
    <div className="enhanced-chatbot-container">
      <div className="chatbot-header">
        <h2>{t('chat.title') || 'Farming Assistant'}</h2>
        <div className="header-actions">
          <button
            className={`voice-btn ${isListening ? 'listening' : ''}`}
            onClick={toggleVoiceInput}
            title={voiceEnabled ? 'Voice Input' : 'Voice not supported'}
            disabled={!voiceEnabled}
          >
            🎤
          </button>
          <select
            value={i18n.language}
            onChange={(e) => i18n.changeLanguage(e.target.value)}
            className="language-select"
          >
            <option value="en">English</option>
            <option value="hi">हिन्दी</option>
            <option value="mr">मराठी</option>
          </select>
        </div>
      </div>

      <div className="messages-container">
        {messages.map((message) => (
          <div key={message.id} className={`message ${message.sender}`}>
            <div className="message-content">
              <p>{message.text}</p>
              {message.sender === 'bot' && message.confidence && (
                <small className="confidence">
                  Confidence: {(message.confidence * 100).toFixed(0)}%
                </small>
              )}
            </div>
            <small className="timestamp">
              {message.timestamp.toLocaleTimeString()}
            </small>
          </div>
        ))}

        {isLoading && (
          <div className="message bot">
            <div className="message-content">
              <p className="loading">Thinking...</p>
            </div>
          </div>
        )}

        {followUpQuestions.length > 0 && (
          <div className="follow-up-questions">
            <h4>Follow-up Questions:</h4>
            {followUpQuestions.map((q) => (
              <div key={q.id} className="question">
                <p>{q.question}</p>
                <div className="options">
                  {q.options?.map((option) => (
                    <button
                      key={option}
                      onClick={() => handleFollowUpQuestion(q.id, option)}
                      className="option-btn"
                    >
                      {option}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {showFeedback && feedbackData && (
          <div className="feedback-section">
            <p>Was this response helpful?</p>
            <div className="feedback-buttons">
              <button
                onClick={() => handleMessageRating(5, true)}
                className="feedback-btn helpful"
              >
                👍 Helpful
              </button>
              <button
                onClick={() => handleMessageRating(1, false)}
                className="feedback-btn not-helpful"
              >
                👎 Not Helpful
              </button>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-section">
        <div className="input-group">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Ask about pests, diseases, or farming..."
            className="chat-input"
            disabled={isLoading}
          />
          <button
            onClick={sendMessage}
            disabled={isLoading || !inputValue.trim()}
            className="send-btn"
          >
            {isLoading ? '...' : 'Send'}
          </button>
        </div>
        <small className="help-text">
          Ask about crop diseases, pest management, spray schedules, or farming practices
        </small>
      </div>
    </div>
  );
};

export default EnhancedChatbot;
