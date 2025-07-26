function doGet(e) {
  try {
    const action = e.parameter.action;
    const ticker = e.parameter.ticker || '';
    const daysBack = parseInt(e.parameter.daysBack) || 7;
    const maxResults = parseInt(e.parameter.maxResults) || 50;
    
    if (action === 'getNewsHeadlines') {
      const headlines = getNewsHeadlines(ticker, daysBack, maxResults);
      return ContentService
        .createTextOutput(JSON.stringify({
          status: 'success',
          headlines: headlines,
          count: headlines.length
        }))
        .setMimeType(ContentService.MimeType.JSON);
    }
    
    return ContentService
      .createTextOutput(JSON.stringify({
        status: 'error',
        error: 'Invalid action'
      }))
      .setMimeType(ContentService.MimeType.JSON);
      
  } catch (error) {
    return ContentService
      .createTextOutput(JSON.stringify({
        status: 'error',
        error: error.toString()
      }))
      .setMimeType(ContentService.MimeType.JSON);
  }
}

function getNewsHeadlines(ticker, daysBack, maxResults) {
  const headlines = [];
  const cutoffDate = new Date();
  cutoffDate.setDate(cutoffDate.getDate() - daysBack);
  
  // Define financial news senders (only Bloomberg NLRT)
  const financialSenders = [
    'nlrt@bloomberg.net'
  ];
  
  // Search for emails from financial news sources with enhanced query support
  let searchQuery = 'from:' + financialSenders[0];
  if (ticker) {
    // Handle both simple ticker searches and complex OR queries
    if (ticker.includes(' OR ')) {
      // For smart search with multiple terms
      searchQuery += ' (' + ticker + ')';
    } else {
      // For simple ticker search
      searchQuery += ' "' + ticker + '"';
    }
  }
  
  try {
    const threads = GmailApp.search(searchQuery, 0, maxResults);
    
    for (let i = 0; i < threads.length && headlines.length < maxResults; i++) {
      const messages = threads[i].getMessages();
      
      for (let j = 0; j < messages.length && headlines.length < maxResults; j++) {
        const message = messages[j];
        const messageDate = message.getDate();
        
        // Skip messages older than cutoff
        if (messageDate < cutoffDate) continue;
        
        const subject = message.getSubject();
        const sender = message.getFrom();
        const snippet = message.getPlainBody().substring(0, 200);
        
        // Check if ticker is mentioned (if specified)
        let tickerMentioned = false;
        if (ticker) {
          const searchText = (subject + ' ' + snippet).toUpperCase();
          tickerMentioned = searchText.includes(ticker.toUpperCase());
        }
        
        headlines.push({
          date: messageDate.toISOString(),
          subject: subject,
          sender: sender,
          snippet: snippet,
          ticker_mentioned: tickerMentioned || !ticker
        });
      }
    }
    
    // Sort by date (newest first)
    headlines.sort((a, b) => new Date(b.date) - new Date(a.date));
    
    return headlines.slice(0, maxResults);
    
  } catch (error) {
    console.error('Error searching Gmail:', error);
    return [];
  }
}

// Test function you can run manually
function testGetNewsHeadlines() {
  const headlines = getNewsHeadlines('AAPL', 7, 10);
  console.log('Found headlines:', headlines.length);
  console.log(headlines);
}