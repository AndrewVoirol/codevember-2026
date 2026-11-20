export const config = {
  matcher: '/',
};

export default function middleware(request: Request): Response {
  const url = new URL(request.url);
  if (url.hostname === 'sea-of-c.andrewvoirol.com') {
    const dest = new URL('/side/sea-of-c/', url);
    return new Response(null, {
      headers: {
        'x-middleware-rewrite': dest.toString(),
      },
    });
  }
  return new Response(null, {
    headers: {
      'x-middleware-next': '1',
    },
  });
}
