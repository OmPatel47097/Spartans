let tp = document.querySelector('button');
tp.addEventListener('click', inputMsg);

function inputMsg() {
    let name = prompt('Enter the name of Student');
    tp.textContent = 'Enter your ID to Enter:' + name;
}