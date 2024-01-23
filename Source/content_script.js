if (window.MutationObserver) {
	var observer = new MutationObserver(function (mutations) {
		Array.prototype.forEach.call(mutations, function (m) {
			if (m.type === 'childList') {
				walk(m.target);
			} else if (m.target.nodeType === 3) {
				handleText(m.target);
			}
		});
	});

	observer.observe(document.body, {
		childList: true,
		attributes: false,
		characterData: true,
		subtree: true
	});
}

walk(document.body);

function walk(node) 
{
	var child, next;
	
	var tagName = node.tagName ? node.tagName.toLowerCase() : "";
	if (tagName == 'input' || tagName == 'textarea') {
		return;
	}
	if (node.classList && node.classList.contains('ace_editor')) {
		return;
	}
	// special bullshit for twitter. this really should be a more general algorithm but its 5 am lol
	if (tagName == 'span') {
		next = node.nextSibling;
		if (next && next.tagName && next.tagName.toLowerCase() == tagName) {
			var next2 = next.nextSibling;
			if (next2 && next2.tagName && next2.tagName.toLowerCase() == tagName) {
				if (next.textContent === " ") {
					var result = handleSpecial(node, next2);
					if (result) {
						return;
					}
				}
			}
		}
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

function handleSpecial(node1, node2)
{
	var tc1 = node1.textContent;
	var tc2 = node2.textContent;
	if (tc1.toLowerCase() === "machine" && tc2.toLowerCase() === "learning") {
		tc1 = tc1.replace(/\bMachine\b/g, "Money");
		tc1 = tc1.replace(/\bmachine\b/g, "money");
		tc2 = tc2.replace(/\bLearning\b/g, "Laundering");
		tc2 = tc2.replace(/\blearning\b/g, "laundering");
		node1.textContent = tc1;
		node2.textContent = tc2;
		return true;
	}
	return false;
}

function handleText(textNode) 
{
	var oldValue = textNode.nodeValue;
	var v = oldValue;

	v = v.replace(/\bMachine Learning\b/g, "Money Laundering");
	v = v.replace(/\bMachine learning\b/g, "Money laundering");
	v = v.replace(/\bmachine Learning\b/g, "money Laundering");
	v = v.replace(/\bmachine learning\b/g, "money laundering");
	
	// avoid infinite series of DOM changes
	if (v !== oldValue) {
		textNode.nodeValue = v;
	}
}


