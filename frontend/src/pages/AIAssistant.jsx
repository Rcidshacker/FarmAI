import React, { useState } from 'react';
import { chatAssistant } from '../services/api';
import { Bot } from 'lucide-react';
// import { NeonButton } from '../components/ui/NeonButton'; // Removed as it is replaced by ChatGPTInput
import ChatGPTInput from '../components/ui/ChatGPTInput';

const AIAssistant = () => {
    const [query, setQuery] = useState("");
    const [messages, setMessages] = useState([
        { type: 'bot', text: 'Namaste! I am your Farm Assistant. Ask me about pests, fertilizers, or current weather conditions.' }
    ]);
    const [loading, setLoading] = useState(false);

    const handleSend = async (text) => {
        // e.preventDefault(); // Not needed as it's not a form event anymore in this context, or handled by component
        // But ChatGPTInput might not pass event, it passes value.
        // If passed from form submit, text is the value.

        const textToSend = typeof text === 'string' ? text : query;
        if (!textToSend.trim()) return;

        const newMessages = [...messages, { type: 'user', text: textToSend }];
        setMessages(newMessages);
        setQuery(""); // Clear local query state if we still use it, though ChatGPTInput manages its own state
        setLoading(true);

        try {
            const res = await chatAssistant(textToSend);
            setMessages([...newMessages, { type: 'bot', text: res.data.response.text }]);
        } catch (error) {
            setMessages([...newMessages, { type: 'bot', text: "Sorry, I encountered an error checking the knowledge base." }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-4xl mx-auto h-[calc(100vh-100px)] flex flex-col mt-6 md:relative">

            <div className="flex-1 overflow-y-auto p-4 pb-32 space-y-6">
                {messages.map((msg, idx) => (
                    <div key={idx} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <div className={`max-w-[80%] rounded-2xl px-5 py-3 ${msg.type === 'user'
                            ? 'bg-black text-white rounded-br-none'
                            : 'bg-white text-gray-800 shadow-sm border border-gray-100 rounded-bl-none'
                            }`}>
                            <p className="whitespace-pre-wrap leading-relaxed">{msg.text}</p>
                        </div>
                    </div>
                ))}
                {loading && <div className="text-gray-400 text-sm ml-4 animate-pulse">Thinking...</div>}
            </div>

            <div className="fixed bottom-[80px] left-0 right-0 px-4 py-3 bg-gradient-to-t from-gray-50 via-gray-50 to-transparent z-40 md:relative md:bottom-0 md:bg-none">
                <div className="max-w-4xl mx-auto">
                    <ChatGPTInput
                        onSubmit={(value) => {
                            setQuery(value);
                            handleSend(value);
                        }}
                        disabled={loading}
                        placeholder="Ask about 'Mealy Bug symptoms' or 'Imidacloprid dosage'..."
                    />
                </div>
            </div>

            {/* Floating Suggestions Button */}
            {!query && messages.length < 3 && (
                <div className="absolute bottom-36 right-4 z-20 md:bottom-24">
                    <button
                        onClick={() => handleSend("What is the weather outlook?")}
                        className="bg-white border border-green-100 shadow-lg shadow-green-900/10 text-green-700 px-4 py-2 rounded-full text-sm font-medium hover:bg-green-50 transition-colors animate-bounce"
                    >
                        🌤️ Weather Outlook?
                    </button>
                </div>
            )}
        </div>
    );
};

export default AIAssistant;
