document.addEventListener("DOMNodeInserted", function(e) {
  walk(document.body);
}, false);

function walk(node) 
{
	// I stole this function from here:
	// http://is.gd/mwZp7E
	
	var child, next;
	
	var tagName = node.tagName ? node.tagName.toLowerCase() : "";
	if (tagName == 'input' || tagName == 'textarea') {
		return;
	}
	if (node.classList && node.classList.contains('ace_editor')) {
		return;
	}

	switch ( node.nodeType )  
	{
		case 1:  // Element
		case 9:  // Document
		case 11: // Document fragment
			child = node.firstChild;
			while ( child ) 
			{
				next = child.nextSibling;
				walk(child);
				child = next;
			}
			break;

		case 3: // Text node
			handleText(node);
			break;
	}
}

function handleText(textNode) 
{
	var v = textNode.nodeValue;

	v = v.replace(/\bMachine Learning\b/g, "Money Laundering");
	v = v.replace(/\bMachine learning\b/g, "Money laundering");
	v = v.replace(/\bmachine Learning\b/g, "money Laundering");
	v = v.replace(/\bmachine learning\b/g, "money laundering");
	
	textNode.nodeValue = v;
}


