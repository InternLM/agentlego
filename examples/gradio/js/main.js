let main_parent = document.getElementById("chat-tab").parentNode;

main_parent.childNodes[0].classList.add("header_bar");
main_parent.style = "padding: 0; margin: 0";
main_parent.parentNode.style = "gap: 0";
main_parent.parentNode.parentNode.style = "padding: 0";

//------------------------------------------------
// Position the chat typing dots
//------------------------------------------------
typing = document.getElementById("typing-container");
typingParent = typing.parentNode;
typingSibling = typing.previousElementSibling;
typingSibling.insertBefore(typing, typingSibling.childNodes[2]);

//------------------------------------------------
// Chat scrolling
//------------------------------------------------
const targetElement = document.getElementById("chat").parentNode.parentNode.parentNode;
targetElement.classList.add("pretty_scrollbar");
targetElement.classList.add("chat-parent");
let isScrolled = false;

targetElement.addEventListener("scroll", function() {
  let diff = targetElement.scrollHeight - targetElement.clientHeight;
  if(Math.abs(targetElement.scrollTop - diff) <= 10 || diff == 0) {
    isScrolled = false;
  } else {
    isScrolled = true;
  }
});

// Create a MutationObserver instance
const observer = new MutationObserver(function(mutations) {
  mutations.forEach(function(mutation) {
    if(!isScrolled) {
      targetElement.scrollTop = targetElement.scrollHeight;
    }

    const firstChild = targetElement.children[0];
    if (firstChild.classList.contains("generating")) {
      typing.parentNode.classList.add("visible-dots");
      document.getElementById("stop").style.display = "flex";
      document.getElementById("Generate").style.display = "none";
    } else {
      typing.parentNode.classList.remove("visible-dots");
      document.getElementById("stop").style.display = "none";
      document.getElementById("Generate").style.display = "flex";
    }

  });
});

// Configure the observer to watch for changes in the subtree and attributes
const config = {
  childList: true,
  subtree: true,
  characterData: true,
  attributeOldValue: true,
  characterDataOldValue: true
};

// Start observing the target element
observer.observe(targetElement, config);

//------------------------------------------------
// Add some scrollbars
//------------------------------------------------
const textareaElements = document.querySelectorAll(".add_scrollbar textarea");
for(i = 0; i < textareaElements.length; i++) {
  textareaElements[i].classList.remove("scroll-hide");
  textareaElements[i].classList.add("pretty_scrollbar");
  textareaElements[i].style.resize = "none";
}

//------------------------------------------------
// Remove some backgrounds
//------------------------------------------------
const noBackgroundelements = document.querySelectorAll(".no-background");
for(i = 0; i < noBackgroundelements.length; i++) {
  noBackgroundelements[i].parentNode.style.border = "none";
  noBackgroundelements[i].parentNode.parentNode.parentNode.style.alignItems = "center";
}

const slimDropdownElements = document.querySelectorAll('.slim-dropdown');
for (i = 0; i < slimDropdownElements.length; i++) {
    const parentNode = slimDropdownElements[i].parentNode;
    parentNode.style.background = 'transparent';
    parentNode.style.border = '0';
}

//------------------------------------------------
// Create the hover menu in the chat tab
// The show/hide events were adapted from:
// https://github.com/SillyTavern/SillyTavern/blob/6c8bd06308c69d51e2eb174541792a870a83d2d6/public/script.js
//------------------------------------------------
var buttonsInChat = document.querySelectorAll("#chat-tab:not(.old-ui) #chat-buttons button");
var button = document.getElementById("hover-element-button");
var menu = document.getElementById("hover-menu");

function showMenu() {
  menu.style.display = "flex"; // Show the menu
}

function hideMenu() {
  menu.style.display = "none"; // Hide the menu
  document.querySelector("#chat-input textarea").focus();
}

if (buttonsInChat.length > 0) {
  for (let i = buttonsInChat.length - 1; i >= 0; i--) {
    const thisButton = buttonsInChat[i];
    menu.appendChild(thisButton);

    thisButton.addEventListener("click", () => {
      hideMenu();
    });

    const buttonText = thisButton.textContent;
    const matches = buttonText.match(/(\(.*?\))/);

    if (matches && matches.length > 1) {
      // Apply the transparent-substring class to the matched substring
      const substring = matches[1];
      const newText = buttonText.replace(substring, `&nbsp;<span class="transparent-substring">${substring.slice(1, -1)}</span>`);
      thisButton.innerHTML = newText;
    }
  }
} else {
  buttonsInChat = document.querySelectorAll("#chat-tab.old-ui #chat-buttons button");
  for (let i = 0; i < buttonsInChat.length; i++) {
    buttonsInChat[i].textContent = buttonsInChat[i].textContent.replace(/ \(.*?\)/, "");
  }
  document.getElementById("gr-hover-container").style.display = "none";
}

function isMouseOverButtonOrMenu() {
  return menu.matches(":hover") || button.matches(":hover");
}

button.addEventListener("mouseenter", function () {
  showMenu();
});

button.addEventListener("click", function () {
  showMenu();
});

// Add event listener for mouseleave on the button
button.addEventListener("mouseleave", function () {
  // Delay to prevent menu hiding when the mouse leaves the button into the menu
  setTimeout(function () {
    if (!isMouseOverButtonOrMenu()) {
      hideMenu();
    }
  }, 100);
});

// Add event listener for mouseleave on the menu
menu.addEventListener("mouseleave", function () {
  // Delay to prevent menu hide when the mouse leaves the menu into the button
  setTimeout(function () {
    if (!isMouseOverButtonOrMenu()) {
      hideMenu();
    }
  }, 100);
});

// Add event listener for click anywhere in the document
document.addEventListener("click", function (event) {
  // Check if the click is outside the button/menu and the menu is visible
  if (!isMouseOverButtonOrMenu() && menu.style.display === "flex") {
    hideMenu();
  }
});

//------------------------------------------------
// Focus on the chat input
//------------------------------------------------
document.querySelector("#chat-input textarea").focus();
