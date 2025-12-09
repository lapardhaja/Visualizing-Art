// simple mobile menu (kept tiny)
const btn = document.querySelector('.menu-toggle');
if (btn) {
    btn.addEventListener('click', () => {
        const expanded = btn.getAttribute('aria-expanded') === 'true';
        btn.setAttribute('aria-expanded', String(!expanded));
        document.body.classList.toggle('menu-open');
    });
}
