import React from 'react';
import { useTranslation } from 'react-i18next';

const LanguageSwitcher = () => {
  const { i18n } = useTranslation();

  const changeLanguage = (lng) => {
    i18n.changeLanguage(lng);
  };

  return (
    <div className="flex gap-2 p-2">
      <button
        onClick={() => changeLanguage('en')}
        className={`px-3 py-1 rounded ${i18n.language === 'en' ? 'bg-green-600 text-white' : 'bg-gray-200'}`}
      >
        English
      </button>
      <button
        onClick={() => changeLanguage('hi')}
        className={`px-3 py-1 rounded ${i18n.language === 'hi' ? 'bg-green-600 text-white' : 'bg-gray-200'}`}
      >
        हिंदी
      </button>
      <button
        onClick={() => changeLanguage('mr')}
        className={`px-3 py-1 rounded ${i18n.language === 'mr' ? 'bg-green-600 text-white' : 'bg-gray-200'}`}
      >
        मराठी
      </button>
    </div>
  );
};

export default LanguageSwitcher;