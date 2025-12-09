import React, { useState, useEffect } from 'react';
import { chatAssistant } from '../services/api';
import { Bot } from 'lucide-react';
// import { NeonButton } from '../components/ui/NeonButton'; // Removed as it is replaced by ChatGPTInput
import ChatGPTInput from '../components/ui/ChatGPTInput';
import { useTranslation } from 'react-i18next';

const AIAssistant = () => {
    const { t } = useTranslation();
    const [query, setQuery] = useState("");
    const [messages, setMessages] = useState([
        { type: 'bot', text: t('assistant.welcome') }
    ]);
    
    // Update welcome message when language changes
    useEffect(() => {
        setMessages(prev => {
            if (prev.length === 1 && prev[0].type === 'bot') {
                return [{ type: 'bot', text: t('assistant.welcome') }];
            }
            return prev;
        });
    }, [t]);

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
            // Handle both response.data.response and direct response.response structures
            const responseText = res.data?.response?.text || res.data?.text || "Unable to get response";
            setMessages([...newMessages, { type: 'bot', text: responseText }]);
        } catch (error) {
            console.error("Assistant error:", error);
            // Provide specific error message
            let errorMsg = t('assistant.errorGeneric');
            if (error.response?.status === 500) {
                errorMsg = t('assistant.errorServer');
            } else if (!navigator.onLine) {
                errorMsg = t('assistant.errorOffline');
            }
            setMessages([...newMessages, { type: 'bot', text: errorMsg }]);
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
